#include <emacs-module.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

int plugin_is_GPL_compatible;

void emacs_signal_error(emacs_env *env, const char* code, const char* message) {
  emacs_value err_sym = env->intern(env, code);
  emacs_value data = env->make_string(env, message, strlen(message));
  env->non_local_exit_signal(env, err_sym, data);
}

int emacs_module_init(struct emacs_runtime* runtime) {
  if (runtime->size < sizeof(*runtime))
    return 1;

  emacs_env* env = runtime->get_environment(runtime);
  if (env->size < sizeof(*env))
    return 2;

  emacs_value provide_fn_args[] = {env->intern(env, "onnx-ml-utils")};
  env->funcall(env, env->intern(env, "provide"), 1, provide_fn_args);

  return 0;
}
