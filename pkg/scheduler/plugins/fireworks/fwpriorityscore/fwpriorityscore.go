// Copyright 2026 Fireworks AI
// SPDX-License-Identifier: Apache-2.0

// Package fwpriorityscore ports Fireworks' priority-aware node scoring into KAI.
//
// Background. multi_pod_manager/pkg/priority_scoring.go serves the /prioritize
// extender verb under enable-priority-aware-scoring (16 clusters). It scores a
// feasible node by how much of it is already held by GPU work of the *same*
// priority class, minus work of a class the scheduling pod cannot displace:
//
//	pref = (samePriorityGPUs + podGPUs - higherPriorityGPUs) / allocatableGPUs
//
// The effect is priority-class homogeneity. Preemptible work clusters together
// and durable deployments stay off nodes already carrying higher-priority work,
// so a deployment's holes are not fragmented by pods it cannot evict.
//
// Why this is not stock. KAI has node scorers, but none of them reads priority:
//
//	gpupack        AddGPUOrderFn -- picks a GPU *index within* a node for
//	               fractional workloads, and returns 0 for whole-GPU pods.
//	               A no-op for Fireworks, which requests whole GPUs.
//	nodeplacement  AddNodeOrderFn -- scores node.NonAllocatedResource, i.e.
//	               free space. Best-fit, with no notion of who holds the rest.
//	priority       AddJobOrderFn only -- queue ordering, not placement.
//
// Pods below the scheduling pod's priority are deliberately ignored: they are
// evictable, so they neither attract nor repel, and binpack already accounts
// for the space they hold.
package fwpriorityscore

import (
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/log"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/plugins/scores"
)

const (
	pluginName = "fwpriorityscore"

	// maxScoreConfig overrides the top of this plugin's score band.
	maxScoreConfig = "maxScore"
)

// defaultMaxScore is the top of the band this plugin scores into.
//
// KAI sums every NodeOrderFn, and the scores package gives each concern its own
// decade: nodeplacement's binpack tops out at scores.MaxHighDensity, resourcetype
// adds a flat scores.ResourceType, and numa, availability, gpusharing and
// topology climb from 100 upward. Segregation has to outrank binpack or best-fit
// silently decides the mixed-class conflicts this plugin exists to settle --
// the same policy MPM encodes by giving /prioritize extender weight 20 against
// NodeResourcesFit's 15. It stays below numa so structural placement still wins.
//
// Setting this at or under scores.MaxHighDensity does not disable the plugin, it
// quietly demotes it to a tie-breaker. MPM carries the same warning in
// priority_scoring.go about never dropping its weight to 15, where the two
// contributions cancel exactly and node choice reverts to a coin flip.
//
// Bounded by node size: preference is a fraction of the node's GPUs, so the
// tightest conflict -- one GPU of difference -- is a gap of 1/alloc and this
// value has to clear scores.MaxHighDensity*alloc. That holds up to 10 GPUs per
// node and fails at 16, so a larger SKU needs this raised (or the score
// weighted per GPU against the largest candidate node) rather than left alone.
const defaultMaxScore = float64(10 * scores.MaxHighDensity)

type fwPriorityScorePlugin struct {
	maxScore float64

	// lowestPref holds, per task, the minimum preference across the nodes that
	// task actually fits on. NodeOrderFn subtracts it so the least-preferred
	// node scores zero while every pairwise gap stays proportional to the real
	// GPU difference -- MPM's shiftPrefsToScores, which subtracts the minimum
	// rather than min-max normalizing for exactly that reason: normalizing
	// would make a one-GPU difference argue as loudly as an eight-GPU one.
	//
	// Written once per task in the pre-order pass and read once per node after,
	// but tasks are not guaranteed to be disjoint in time, so it is guarded.
	mu         sync.RWMutex
	lowestPref map[string]float64
}

func New(arguments framework.PluginArguments) framework.Plugin {
	plugin := &fwPriorityScorePlugin{
		maxScore:   defaultMaxScore,
		lowestPref: make(map[string]float64),
	}

	if raw, ok := arguments[maxScoreConfig]; ok {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
		switch {
		case err != nil:
			log.InfraLogger.Errorf("%s: failed to parse %s=%q: %v, keeping default %v",
				pluginName, maxScoreConfig, raw, err, plugin.maxScore)
		case parsed <= 0:
			log.InfraLogger.Errorf("%s: %s=%v is not positive, keeping default %v",
				pluginName, maxScoreConfig, parsed, plugin.maxScore)
		default:
			if parsed <= scores.MaxHighDensity {
				log.InfraLogger.Errorf(
					"%s: %s=%v is at or below binpack's ceiling (%v); segregation will only break ties",
					pluginName, maxScoreConfig, parsed, float64(scores.MaxHighDensity))
			}
			plugin.maxScore = parsed
		}
	}

	return plugin
}

func (p *fwPriorityScorePlugin) Name() string {
	return pluginName
}

func (p *fwPriorityScorePlugin) OnSessionOpen(ssn *framework.Session) {
	p.mu.Lock()
	p.lowestPref = make(map[string]float64)
	p.mu.Unlock()

	ssn.AddNodePreOrderFn(p.nodePreOrderFn)
	ssn.AddNodeOrderFn(p.nodeOrderFn)
}

func (p *fwPriorityScorePlugin) OnSessionClose(_ *framework.Session) {}

// nodePreOrderFn records the task's least-preferred fitting node. The shift in
// nodeOrderFn needs a minimum over the whole candidate set, which a per-node
// NodeOrderFn cannot see; this is the same reason nodeplacement's binpack
// registers a pre-order pass to find its min/max allocatable.
func (p *fwPriorityScorePlugin) nodePreOrderFn(task *pod_info.PodInfo, fittingNodes []*node_info.NodeInfo) error {
	podGPUs := taskGPUs(task)
	if podGPUs == 0 {
		return nil
	}

	lowest := math.Inf(1)
	for _, node := range fittingNodes {
		pref, ok := nodePreference(task, node, podGPUs)
		if ok && pref < lowest {
			lowest = pref
		}
	}
	if math.IsInf(lowest, 1) {
		return nil // no fitting node advertises GPUs; leave every score neutral
	}

	p.mu.Lock()
	p.lowestPref[string(task.UID)] = lowest
	p.mu.Unlock()
	return nil
}

func (p *fwPriorityScorePlugin) nodeOrderFn(task *pod_info.PodInfo, node *node_info.NodeInfo) (float64, error) {
	podGPUs := taskGPUs(task)
	if podGPUs == 0 {
		return 0, nil // CPU-only work has no priority-class GPU story
	}

	p.mu.RLock()
	lowest, ok := p.lowestPref[string(task.UID)]
	p.mu.RUnlock()
	if !ok {
		return 0, nil
	}

	pref, scorable := nodePreference(task, node, podGPUs)
	if !scorable {
		return 0, nil
	}

	score := p.maxScore * (pref - lowest)
	if score < 0 {
		score = 0 // unreachable by construction; guards float rounding
	}
	if score > p.maxScore {
		score = p.maxScore
	}

	log.InfraLogger.V(7).Do(func() {
		log.InfraLogger.Infof(
			"Estimating Task: <%v/%v> Job: <%v> for node: <%s>. Preference: %f, Score: %f",
			task.Namespace, task.Name, task.Job, node.Name, pref, score)
	})
	return score, nil
}

// nodePreference is MPM's priorityPref: the pull of same-class work plus this
// pod, minus the penalty for work that outranks it, as a fraction of the node's
// GPUs. The result lies in [-1, 1]. ok is false when the node advertises no
// GPUs, so it cannot be scored and must not drag the minimum down.
func nodePreference(task *pod_info.PodInfo, node *node_info.NodeInfo, podGPUs float64) (float64, bool) {
	alloc := float64(node.GetNumberOfGPUsInNode())
	if alloc <= 0 {
		return 0, false
	}

	pri := taskPriority(task)
	var samePriority, higherPriority float64
	for _, resident := range node.PodInfos {
		// Terminal and unbound pods hold no GPUs. AllocatedStatus is KAI's own
		// answer to the Succeeded/Failed/deleting checks MPM makes by hand.
		if resident == nil || !pod_status.AllocatedStatus(resident.Status) {
			continue
		}
		if resident.UID == task.UID {
			continue
		}
		gpus := taskGPUs(resident)
		if gpus == 0 {
			continue
		}
		switch residentPriority := taskPriority(resident); {
		case residentPriority == pri:
			samePriority += gpus
		case residentPriority > pri:
			higherPriority += gpus
		}
	}

	return (samePriority + podGPUs - higherPriority) / alloc, true
}

// taskGPUs counts whole and DRA GPUs the same way PodInfo builds its own
// resource vector, so a DRA-backed pod is not silently treated as CPU-only.
func taskGPUs(task *pod_info.PodInfo) float64 {
	if task == nil {
		return 0
	}
	return task.GpuRequirement.GPUs() + float64(task.GpuRequirement.GetDraGpusCount())
}

// taskPriority mirrors MPM's podPriorityValue: an unset priority is 0.
func taskPriority(task *pod_info.PodInfo) int32 {
	if task == nil || task.Pod == nil || task.Pod.Spec.Priority == nil {
		return 0
	}
	return *task.Pod.Spec.Priority
}
