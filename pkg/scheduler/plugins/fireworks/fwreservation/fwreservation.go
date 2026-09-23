// Copyright 2026 Fireworks AI
// SPDX-License-Identifier: Apache-2.0

// Package fwreservation ports Fireworks' reservation-ownership eviction rules
// into KAI.
//
// Background. In the Fireworks fleet, GPU capacity is partitioned by a node
// taint (fireworks.ai/reservation=<pool>, plus fireworks.ai/rftj on clusters
// that dedicate training capacity). Two kinds of pod tolerate it, and the
// *operator* of the toleration is what distinguishes them — see
// control_plane/pkg/k8s/scheduling_extra_values.go:
//
//	owner     Equal + a specific value, with node affinity In [value].
//	          Bound to exactly one pool.
//	borrower  Exists with no value. Tolerates any pool's taint, so it lands
//	          wherever a reservation is currently idle.
//
// Borrowers also run strictly below the owner tier (PriorityValueDeployment = 2;
// SpotProd 1, SpotDev 0, BIJCPU -1, BIJ -5), which is what makes them evictable.
//
// multi_pod_manager/pkg/reclaim.go encodes the same rule as isReclaimableBorrower
// and drives it from a kube-scheduler extender: it marks non-candidate nodes
// UnschedulableAndUnresolvable so kube-scheduler's own preemption aims at the
// reserved, reclaimable nodes instead of spilling the owner onto free capacity.
//
// KAI has no extender mechanism, but it does have the hook that logic actually
// wants: a victim filter. This plugin supplies it directly, which is a better
// fit than steering someone else's preemption by hiding nodes from it.
//
// Scope. This plugin only ever *narrows* eviction: it answers "may this victim
// be taken for this pending job" and returns false to veto. It never widens the
// candidate set, so stacking it with the stock plugins is safe.
//
// Not ported (deliberately, for now):
//   - NVLink/RDMA partition scope (PlacementScope.SelectedNVLink and friends).
//     KAI's topology plugin already constrains placement on the fabric labels.
//   - DRA claim eviction (reclaim_dra_eviction.go). KAI has its own
//     dynamicresources plugin in both scheduler and binder.
//   - exclusiveNodeTaintValues. That list exists because kube-scheduler
//     preemption cannot fire when a pool runs below the deployment tier; KAI's
//     preempt action may not share that limitation, so it needs measuring
//     before being ported.
package fwreservation

import (
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/podgroup_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/log"
)

const (
	pluginName = "fwreservation"

	// reservationTaintKeysConfig is a comma-separated list of taint keys that
	// designate reservation ownership. Mirrors normalizeReservationTaintKeys in
	// multi_pod_manager.
	reservationTaintKeysConfig = "reservationTaintKeys"

	// ownerPriorityConfig is the priority at or above which a pod is treated as
	// a non-preemptible owner. Matches reclaimOwnerPriority in
	// multi_pod_manager/pkg/reclaim.go, which tracks
	// constants.PriorityValueDeployment.
	ownerPriorityConfig = "ownerPriority"

	defaultReservationTaintKey = "fireworks.ai/reservation"
	defaultOwnerPriority       = int32(2)
)

type fwReservationPlugin struct {
	// reservationKeys are the taint keys treated as reservation ownership.
	reservationKeys []string
	// ownerPriority is the fallback owner threshold, used when the pending job
	// carries no priority of its own.
	ownerPriority int32
}

func New(arguments framework.PluginArguments) framework.Plugin {
	plugin := &fwReservationPlugin{
		reservationKeys: []string{defaultReservationTaintKey},
		ownerPriority:   defaultOwnerPriority,
	}

	if raw, ok := arguments[reservationTaintKeysConfig]; ok {
		var keys []string
		for _, k := range strings.Split(raw, ",") {
			if k = strings.TrimSpace(k); k != "" {
				keys = append(keys, k)
			}
		}
		if len(keys) > 0 {
			plugin.reservationKeys = keys
		} else {
			log.InfraLogger.Errorf("%s: %s was set but parsed to no keys, keeping default %v",
				pluginName, reservationTaintKeysConfig, plugin.reservationKeys)
		}
	}

	if raw, ok := arguments[ownerPriorityConfig]; ok {
		parsed, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 32)
		if err != nil {
			log.InfraLogger.Errorf("%s: failed to parse %s=%q: %v, keeping default %d",
				pluginName, ownerPriorityConfig, raw, err, plugin.ownerPriority)
		} else {
			plugin.ownerPriority = int32(parsed)
		}
	}

	return plugin
}

func (p *fwReservationPlugin) Name() string {
	return pluginName
}

func (p *fwReservationPlugin) OnSessionOpen(ssn *framework.Session) {
	// Both actions get the same rule. Fireworks' "reclaim" is priority-based
	// eviction scoped by pool ownership, which lines up with KAI's preempt
	// action; registering on reclaim as well keeps the semantics identical if
	// queues are later given real quota and the reclaim action starts firing.
	ssn.AddPreemptVictimFilterFn(p.victimFilterFn)
	ssn.AddReclaimVictimFilterFn(p.victimFilterFn)
}

func (p *fwReservationPlugin) OnSessionClose(_ *framework.Session) {}

// victimFilterFn reports whether victim may be evicted to make room for
// pendingJob.
//
// It vetoes only when pendingJob is a reservation owner and victim is not a
// borrower of that same pool. Any other combination is left to the other
// plugins, so this plugin is additive.
func (p *fwReservationPlugin) victimFilterFn(pendingJob *podgroup_info.PodGroupInfo, victim *podgroup_info.PodGroupInfo) bool {
	if pendingJob == nil || victim == nil {
		return true
	}

	ownerKey, isOwner := p.ownedPool(pendingJob)
	if !isOwner {
		// Not a reservation owner: this plugin has no opinion. A borrower
		// evicting another borrower, or ordinary non-reserved work, is
		// somebody else's decision.
		return true
	}

	threshold := pendingJob.Priority
	if threshold == 0 {
		threshold = p.ownerPriority
	}

	if victim.Priority >= threshold {
		// Same tier or above: an owner, or another protected workload. MPM's
		// rule is strictly-below, and that strictness is load-bearing — it is
		// exactly why exclusiveNodeTaintValues has to exist for pools running
		// beneath the deployment tier.
		return false
	}

	if !p.toleratesPoolWithExists(victim, ownerKey) {
		// Below the threshold but not a borrower of this pool — it is not
		// squatting on the owner's reservation, so evicting it would free
		// capacity the owner cannot use anyway.
		return false
	}

	return true
}

// ownedPool returns the reservation key a job owns, via an Equal toleration
// carrying a specific value. Returns false when the job owns no pool, which
// includes borrowers (Exists, no value).
func (p *fwReservationPlugin) ownedPool(job *podgroup_info.PodGroupInfo) (string, bool) {
	for _, pod := range job.GetAllPodsMap() {
		if pod == nil || pod.Pod == nil {
			continue
		}
		for _, tol := range pod.Pod.Spec.Tolerations {
			if tol.Operator != v1.TolerationOpEqual || tol.Value == "" {
				continue
			}
			for _, key := range p.reservationKeys {
				if tol.Key == key {
					return key, true
				}
			}
		}
	}
	return "", false
}

// toleratesPoolWithExists reports whether the job is a borrower of the given
// reservation key: an Exists toleration with no named value, which is how
// scheduling_extra_values.go stamps enableBorrowing workloads.
func (p *fwReservationPlugin) toleratesPoolWithExists(job *podgroup_info.PodGroupInfo, key string) bool {
	for _, pod := range job.GetAllPodsMap() {
		if pod == nil || pod.Pod == nil {
			continue
		}
		for _, tol := range pod.Pod.Spec.Tolerations {
			if tol.Key == key && tol.Operator == v1.TolerationOpExists {
				return true
			}
		}
	}
	return false
}
