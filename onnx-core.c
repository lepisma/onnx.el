#include <emacs-module.h>
#include <string.h>
#include <onnxruntime_c_api.h>

int plugin_is_GPL_compatible;
const char *onnx_core_version = "0.0.1";

static emacs_value Fonnx_core_version(emacs_env *env, ptrdiff_t n, emacs_value args[], void *data) {
  return env->make_string(env, onnx_core_version, strlen(onnx_core_version));
}

int emacs_module_init(struct emacs_runtime *runtime) {
  if (runtime->size < sizeof (*runtime))
    return 1;

  emacs_env *env = runtime->get_environment(runtime);
  if (env->size < sizeof(*env))
    return 2;

  emacs_value version_fn = env->make_function(env, 0, 0, Fonnx_core_version, "Return version of onnx-core.", NULL);
  emacs_value version_fn_sym = env->intern(env, "onnx-core-version");
  emacs_value version_fn_args[] = {version_fn_sym, version_fn};

  env->funcall(env, env->intern(env, "defalias"), 2, version_fn_args);

  return 0;
}
