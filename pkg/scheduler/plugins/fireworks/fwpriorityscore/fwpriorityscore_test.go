// Copyright 2026 Fireworks AI
// SPDX-License-Identifier: Apache-2.0

package fwpriorityscore

import (
	"math"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/common_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/resource_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/cache"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/cache/cluster_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/plugins/scores"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/nodes_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/resources_fake"
)

const (
	deploymentPriority = int32(2) // PriorityValueDeployment
	spotProdPriority   = int32(1)
	bijPriority        = int32(-5)
)

// testVectorMap covers every resource the fakes below put on a node or a pod,
// so PodInfo and NodeInfo index into the same vector space.
var testVectorMap = func() *resource_info.ResourceVectorMap {
	m := resource_info.NewResourceVectorMap()
	gpus := "8"
	for name := range *resources_fake.BuildResourceList(nil, nil, &gpus, nil) {
		m.AddResource(name)
	}
	m.AddResource(resource_info.GPUResourceName)
	return m
}()

// task builds a PodInfo requesting gpus at the given priority. Only priority,
// GPU count and status are read by the scorer.
func task(uid string, priority int32, gpus string) *pod_info.PodInfo {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: uid, Namespace: "default"},
		Spec: v1.PodSpec{
			Priority: &priority,
			Containers: []v1.Container{{
				Resources: v1.ResourceRequirements{
					Requests: *resources_fake.BuildResourceList(nil, nil, &gpus, nil),
					Limits:   *resources_fake.BuildResourceList(nil, nil, &gpus, nil),
				},
			}},
		},
	}
	pi := pod_info.NewTaskInfo(pod, testVectorMap)
	pi.UID = common_info.PodID(uid)
	pi.Status = pod_status.Running
	return pi
}

// node builds an 8-GPU NodeInfo already holding residents.
func node(name string, residents ...*pod_info.PodInfo) *node_info.NodeInfo {
	gpus := "8"
	res := resources_fake.BuildResourceList(nil, nil, &gpus, nil)
	n := nodes_fake.BuildNode(name, res, res)

	affinity := cluster_info.NewK8sNodePodAffinityInfo(n, cache.NewK8sClusterPodAffinityInfo())
	ni := node_info.NewNodeInfo(n, affinity, testVectorMap)

	for _, r := range residents {
		ni.PodInfos[r.UID] = r
	}
	return ni
}

// scoreAll runs the real two-pass flow: pre-order over every candidate, then
// one score per node. Scoring without the pre-order is a separate test.
func scoreAll(t *testing.T, p *fwPriorityScorePlugin, pending *pod_info.PodInfo, nodes ...*node_info.NodeInfo) map[string]float64 {
	t.Helper()
	if err := p.nodePreOrderFn(pending, nodes); err != nil {
		t.Fatalf("pre-order: %v", err)
	}
	out := make(map[string]float64, len(nodes))
	for _, n := range nodes {
		score, err := p.nodeOrderFn(pending, n)
		if err != nil {
			t.Fatalf("score %s: %v", n.Name, err)
		}
		out[n.Name] = score
	}
	return out
}

func newPlugin() *fwPriorityScorePlugin {
	return &fwPriorityScorePlugin{maxScore: defaultMaxScore, lowestPref: map[string]float64{}}
}

// The core policy: a node already carrying same-class work outranks an empty
// one, so same-priority work concentrates instead of spreading.
func TestSamePriorityConcentrates(t *testing.T) {
	pending := task("pending", deploymentPriority, "2")
	occupied := node("occupied", task("peer", deploymentPriority, "4"))
	empty := node("empty")

	got := scoreAll(t, newPlugin(), pending, occupied, empty)

	if got["occupied"] <= got["empty"] {
		t.Fatalf("same-class node must outrank empty: occupied=%v empty=%v", got["occupied"], got["empty"])
	}
	if got["empty"] != 0 {
		t.Errorf("least-preferred node anchors at zero, got %v", got["empty"])
	}
}

// The other half: work the pending pod cannot displace repels it, so a
// deployment does not land on a node whose holes it can never reclaim.
func TestHigherPriorityRepels(t *testing.T) {
	pending := task("pending", spotProdPriority, "2")
	blocked := node("blocked", task("boss", deploymentPriority, "4"))
	empty := node("empty")

	got := scoreAll(t, newPlugin(), pending, blocked, empty)

	if got["blocked"] >= got["empty"] {
		t.Fatalf("higher-class node must rank below empty: blocked=%v empty=%v", got["blocked"], got["empty"])
	}
	if got["blocked"] != 0 {
		t.Errorf("least-preferred node anchors at zero, got %v", got["blocked"])
	}
}

// Lower-priority residents are evictable, so they neither attract nor repel.
// Binpack already accounts for the space they hold.
func TestLowerPriorityIgnored(t *testing.T) {
	pending := task("pending", deploymentPriority, "2")
	withBIJ := node("with-bij", task("bij", bijPriority, "4"))
	empty := node("empty")

	got := scoreAll(t, newPlugin(), pending, withBIJ, empty)

	if got["with-bij"] != got["empty"] {
		t.Fatalf("lower-class work must not move the score: with-bij=%v empty=%v", got["with-bij"], got["empty"])
	}
}

// The policy is only real if it outranks binpack. MPM encodes this as extender
// weight 20 against NodeResourcesFit's 15; here the band has to clear
// scores.MaxHighDensity, which is the most nodeplacement can ever contribute.
func TestSegregationOutranksBinpack(t *testing.T) {
	pending := task("pending", deploymentPriority, "2")
	occupied := node("occupied", task("peer", deploymentPriority, "4"))
	empty := node("empty")

	got := scoreAll(t, newPlugin(), pending, occupied, empty)

	gap := got["occupied"] - got["empty"]
	if gap <= float64(scores.MaxHighDensity) {
		t.Fatalf("gap %v cannot outrank binpack's ceiling %v; best-fit would decide mixed-class conflicts",
			gap, float64(scores.MaxHighDensity))
	}
}

// Gaps stay proportional to the real GPU difference. Min-max normalization
// would stretch every non-tie to the full band, making a one-GPU difference
// argue as loudly as a four-GPU one -- which is why MPM subtracts the minimum
// instead. Two same-class GPUs against four must score exactly half.
func TestGapsStayProportional(t *testing.T) {
	pending := task("pending", deploymentPriority, "1")
	four := node("four", task("a", deploymentPriority, "4"))
	two := node("two", task("b", deploymentPriority, "2"))
	empty := node("empty")

	got := scoreAll(t, newPlugin(), pending, four, two, empty)

	if math.Abs(got["two"]*2-got["four"]) > 1e-9 {
		t.Fatalf("expected two=%v to be exactly half of four=%v", got["two"], got["four"])
	}
}

// A CPU-only pod has no priority-class GPU story, so it must not perturb the
// sum every other scorer contributes to.
func TestCPUOnlyTaskScoresNeutral(t *testing.T) {
	pending := task("pending", deploymentPriority, "0")
	occupied := node("occupied", task("peer", deploymentPriority, "4"))

	got := scoreAll(t, newPlugin(), pending, occupied)

	if got["occupied"] != 0 {
		t.Fatalf("CPU-only task must score neutral, got %v", got["occupied"])
	}
}

// Scoring a task the pre-order never saw must be neutral rather than scored
// against a stale or zero minimum, which would invent a preference.
func TestScoringWithoutPreOrderIsNeutral(t *testing.T) {
	p := newPlugin()
	pending := task("pending", deploymentPriority, "2")

	score, err := p.nodeOrderFn(pending, node("occupied", task("peer", deploymentPriority, "4")))
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if score != 0 {
		t.Fatalf("no pre-order means no minimum to shift by; expected 0, got %v", score)
	}
}

// Terminal residents released their GPUs, so they are not occupancy.
func TestTerminalResidentsIgnored(t *testing.T) {
	pending := task("pending", deploymentPriority, "2")

	dead := task("dead", deploymentPriority, "4")
	dead.Status = pod_status.Succeeded
	withDead := node("with-dead", dead)
	empty := node("empty")

	got := scoreAll(t, newPlugin(), pending, withDead, empty)

	if got["with-dead"] != got["empty"] {
		t.Fatalf("terminal resident must not count as occupancy: with-dead=%v empty=%v",
			got["with-dead"], got["empty"])
	}
}

func TestNewParsesMaxScore(t *testing.T) {
	tests := []struct {
		name string
		args map[string]string
		want float64
	}{
		{name: "default", args: nil, want: defaultMaxScore},
		{name: "override", args: map[string]string{maxScoreConfig: "45"}, want: 45},
		{name: "whitespace tolerated", args: map[string]string{maxScoreConfig: " 45 "}, want: 45},
		{name: "unparseable keeps default", args: map[string]string{maxScoreConfig: "high"}, want: defaultMaxScore},
		{name: "zero keeps default", args: map[string]string{maxScoreConfig: "0"}, want: defaultMaxScore},
		{name: "negative keeps default", args: map[string]string{maxScoreConfig: "-5"}, want: defaultMaxScore},
		// Accepted, but demoted to a tie-breaker; New logs about it.
		{name: "below binpack ceiling is accepted", args: map[string]string{maxScoreConfig: "5"}, want: 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := New(tt.args).(*fwPriorityScorePlugin)
			if p.maxScore != tt.want {
				t.Fatalf("maxScore = %v, want %v", p.maxScore, tt.want)
			}
		})
	}
}

func TestScoreNeverExceedsBand(t *testing.T) {
	pending := task("pending", deploymentPriority, "8")
	full := node("full", task("peer", deploymentPriority, "8"))
	blocked := node("blocked", task("boss", int32(99), "8"))

	got := scoreAll(t, newPlugin(), pending, full, blocked)

	for name, score := range got {
		if score < 0 || score > defaultMaxScore {
			t.Errorf("%s scored %v, outside [0, %v]", name, score, defaultMaxScore)
		}
	}
}
