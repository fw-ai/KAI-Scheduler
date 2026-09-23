// Copyright 2026 Fireworks AI
// SPDX-License-Identifier: Apache-2.0

package plugins

import (
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/plugins/fireworks/fwpriorityscore"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/plugins/fireworks/fwreclaims"
)

// Fireworks plugins register themselves here rather than from InitDefaultPlugins
// in factory.go.
//
// Why a separate file and an init(): factory.go is the one file every upstream
// release touches — each new upstream plugin appends to the same list — so
// editing it guarantees a conflict on every rebase. This file is one upstream
// will never create, and framework.RegisterPluginBuilder is additive:
// pluginBuilders is a package-level map that InitDefaultPlugins only writes to,
// never resets, and the mutex makes ordering irrelevant. So an init() here is
// equivalent to a line in factory.go with none of the merge cost.
//
// Adding a plugin: create pkg/scheduler/plugins/fireworks/<name>/, then add one
// line below. Nothing upstream owns is edited, so `git rebase --onto <newtag>`
// stays conflict-free.
func init() {
	framework.RegisterPluginBuilder("fwreclaims", fwreclaims.New)
	framework.RegisterPluginBuilder("fwpriorityscore", fwpriorityscore.New)
}
