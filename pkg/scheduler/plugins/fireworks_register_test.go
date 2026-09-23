package plugins

import (
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

// The init() in fireworks_register.go must register without InitDefaultPlugins
// having run, and must survive it running afterwards.
func TestFireworksPluginRegisteredByInit(t *testing.T) {
	if _, ok := framework.GetPluginBuilder("fwreclaims"); !ok {
		t.Fatal("fwreclaims not registered by init()")
	}
	InitDefaultPlugins()
	if _, ok := framework.GetPluginBuilder("fwreclaims"); !ok {
		t.Fatal("InitDefaultPlugins clobbered the fireworks registration")
	}
	if _, ok := framework.GetPluginBuilder("predicates"); !ok {
		t.Fatal("stock plugins missing after InitDefaultPlugins")
	}
}
