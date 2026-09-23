// Copyright 2026 Fireworks AI
// SPDX-License-Identifier: Apache-2.0

package fwreclaims

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/common_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/podgroup_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

const resKey = "fireworks.ai/reservation"

// job builds a single-pod PodGroupInfo carrying the given priority and
// tolerations — enough for the victim filter, which only reads those two.
func job(uid string, priority int32, tols ...v1.Toleration) *podgroup_info.PodGroupInfo {
	task := &pod_info.PodInfo{
		UID:    common_info.PodID(uid + "-task"),
		Status: pod_status.Running,
		Pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: uid, Namespace: "default"},
			Spec:       v1.PodSpec{Tolerations: tols},
		},
	}
	pg := podgroup_info.NewPodGroupInfo(common_info.PodGroupID(uid), task)
	pg.Priority = priority
	return pg
}

// owner tolerates one named pool: Equal + value, per scheduling_extra_values.go.
func ownerTol(value string) v1.Toleration {
	return v1.Toleration{
		Key:      resKey,
		Operator: v1.TolerationOpEqual,
		Value:    value,
		Effect:   v1.TaintEffectNoSchedule,
	}
}

// borrower tolerates any pool: Exists, no value.
func borrowerTol() v1.Toleration {
	return v1.Toleration{
		Key:      resKey,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}
}

func newPlugin(t *testing.T, args framework.PluginArguments) *fwReclaimsPlugin {
	t.Helper()
	p, ok := New(args).(*fwReclaimsPlugin)
	if !ok {
		t.Fatalf("New() did not return *fwReclaimsPlugin")
	}
	return p
}

func TestVictimFilter(t *testing.T) {
	const (
		deployment = int32(2) // PriorityValueDeployment
		spotProd   = int32(1)
		spotDev    = int32(0)
		bij        = int32(-5)
	)

	cases := map[string]struct {
		pending *podgroup_info.PodGroupInfo
		victim  *podgroup_info.PodGroupInfo
		want    bool
	}{
		"owner evicts borrower below its tier": {
			pending: job("owner", deployment, ownerTol("acme")),
			victim:  job("borrower", spotProd, borrowerTol()),
			want:    true,
		},
		"owner evicts batch borrower far below": {
			pending: job("owner", deployment, ownerTol("acme")),
			victim:  job("bij", bij, borrowerTol()),
			want:    true,
		},
		"owner may not evict another owner at the same tier": {
			pending: job("owner", deployment, ownerTol("acme")),
			victim:  job("other-owner", deployment, ownerTol("contoso")),
			want:    false,
		},
		// The strictly-below rule. This is precisely the case
		// exclusiveNodeTaintValues exists for: a pool running at the squatter's
		// tier can never evict it, so the watcher declines to claim the node.
		"owner may not evict a borrower at its own tier": {
			pending: job("owner", spotProd, ownerTol("acme")),
			victim:  job("squatter", spotProd, borrowerTol()),
			want:    false,
		},
		"owner may not evict a non-borrower below its tier": {
			pending: job("owner", deployment, ownerTol("acme")),
			victim:  job("unrelated", spotDev),
			want:    false,
		},
		// A pod pinned to a different pool by Equal is an owner there, not a
		// borrower here, so it is not squatting on our reservation.
		"owner may not evict a lower-tier owner of another pool": {
			pending: job("owner", deployment, ownerTol("acme")),
			victim:  job("other-pool", spotDev, ownerTol("contoso")),
			want:    false,
		},
		// Plugin is additive: with no ownership claim it abstains and lets the
		// stock plugins decide.
		"non-owner pending job: no opinion": {
			pending: job("borrower", spotProd, borrowerTol()),
			victim:  job("other", spotDev, borrowerTol()),
			want:    true,
		},
		"pending job with no tolerations: no opinion": {
			pending: job("plain", deployment),
			victim:  job("other", spotDev, borrowerTol()),
			want:    true,
		},
		"nil victim is not vetoed": {
			pending: job("owner", deployment, ownerTol("acme")),
			victim:  nil,
			want:    true,
		},
	}

	p := newPlugin(t, framework.PluginArguments{})
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := p.victimFilterFn(tc.pending, tc.victim); got != tc.want {
				t.Errorf("victimFilterFn = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestOwnerPriorityFallback pins a quirk inherited from MPM rather than a
// designed behaviour.
//
// reclaim.go falls back to reclaimOwnerPriority when the owner's priority is
// zero — but zero is also SpotDev, a real tier, not just Go's zero value. So an
// owner running at SpotDev is treated as though it were at the deployment tier
// and can evict SpotProd(1) and SpotDev(0) borrowers, even though neither is
// strictly below it.
//
// This test asserts the current behaviour so the port stays faithful. If that
// conflation is ever fixed upstream in MPM, this is the test that should change
// with it.
func TestOwnerPriorityFallback(t *testing.T) {
	p := newPlugin(t, framework.PluginArguments{})
	pending := job("owner", 0, ownerTol("acme"))

	if got := p.victimFilterFn(pending, job("spotdev", 0, borrowerTol())); !got {
		t.Error("priority-0 owner falls back to threshold 2, so a spotDev borrower is evictable")
	}
	if got := p.victimFilterFn(pending, job("bij", -5, borrowerTol())); !got {
		t.Error("victim below the fallback threshold should be evictable")
	}
	// An owner that is genuinely at the deployment tier is unaffected.
	if got := p.victimFilterFn(job("owner2", 2, ownerTol("acme")), job("peer", 2, borrowerTol())); got {
		t.Error("same-tier victim must not be evictable when the owner has a real priority")
	}
}

func TestConfiguration(t *testing.T) {
	t.Run("custom keys", func(t *testing.T) {
		p := newPlugin(t, framework.PluginArguments{
			reservationTaintKeysConfig: "fireworks.ai/reservation, fireworks.ai/rftj",
		})
		if len(p.reservationKeys) != 2 {
			t.Fatalf("reservationKeys = %v, want 2 entries", p.reservationKeys)
		}
		rftj := v1.Toleration{Key: "fireworks.ai/rftj", Operator: v1.TolerationOpEqual, Value: "team", Effect: v1.TaintEffectNoSchedule}
		borrow := v1.Toleration{Key: "fireworks.ai/rftj", Operator: v1.TolerationOpExists, Effect: v1.TaintEffectNoSchedule}
		if got := p.victimFilterFn(job("o", 2, rftj), job("b", 1, borrow)); !got {
			t.Error("rftj owner should be able to evict an rftj borrower")
		}
	})

	t.Run("custom owner priority", func(t *testing.T) {
		p := newPlugin(t, framework.PluginArguments{ownerPriorityConfig: "10"})
		if p.ownerPriority != 10 {
			t.Fatalf("ownerPriority = %d, want 10", p.ownerPriority)
		}
	})

	t.Run("bad values keep defaults", func(t *testing.T) {
		p := newPlugin(t, framework.PluginArguments{
			ownerPriorityConfig:        "not-a-number",
			reservationTaintKeysConfig: " , ",
		})
		if p.ownerPriority != defaultOwnerPriority {
			t.Errorf("ownerPriority = %d, want default %d", p.ownerPriority, defaultOwnerPriority)
		}
		if len(p.reservationKeys) != 1 || p.reservationKeys[0] != defaultReservationTaintKey {
			t.Errorf("reservationKeys = %v, want default", p.reservationKeys)
		}
	})
}
