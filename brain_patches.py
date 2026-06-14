# brain_patches.py — live hot-patch extension point for the Alpha engine.
#
# BrainPatcher (in brain.py) imports this module roughly every 2.5s, but only
# when the file's mtime changes, and then calls:
#
#       apply_patches(alpha_brain, shared_sem)
#
# Use it to monkeypatch a RUNNING NeuromorphicBrain instance (tune constants,
# swap a method, recalibrate the amygdala, …) without rebuilding the binary.
#
# It is empty by default. The two-personality tuning patches that used to live
# here were folded into brain.py and are no longer needed for the single-brain
# Alpha engine.

import gc


def _find_class(module_obj, name):
    """Best-effort: locate a class object by name from any object that carries
    a reference to the brain module's globals (e.g. an instance's class module)."""
    try:
        mod = __import__(getattr(type(module_obj), "__module__", "brain"))
        return getattr(mod, name, None)
    except Exception:
        return None


def _find_host(NB):
    """Locate the running NeuromorphicBrain instance (one-time gc scan)."""
    if NB is None:
        return None
    for o in gc.get_objects():
        try:
            if isinstance(o, NB):
                return o
        except Exception:
            pass
    return None


def apply_patches(alpha_brain, shared_sem):
    """No-op by default. Add live patches to the running Alpha brain here.

    `alpha_brain` is Alpha's brain object (the single AlphaBrain instance);
    `shared_sem` is the SharedSemanticDictionary. Return value is ignored.
    """
    return
