#include <emacs-module.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

int plugin_is_GPL_compatible;
const double EPS = 1E-12;

void emacs_signal_error(emacs_env *env, const char* code, const char* message) {
  emacs_value err_sym = env->intern(env, code);
  emacs_value data = env->make_string(env, message, strlen(message));
  env->non_local_exit_signal(env, err_sym, data);
}

double compute_l2_norm(emacs_env *env, emacs_value vector) {
  size_t len = env->vec_size(env, vector);

  double output = 0;
  for (size_t i = 0; i < len; i++) {
    output = output + pow(env->extract_float(env, env->vec_get(env, vector, i)), 2);
  }

  return sqrt(output);
}

/*
 * Normalize given numerical single dimensional vector using L2 norm
 * destructively.
 */
void nl2_normalize(emacs_env* env, emacs_value vector) {
  double norm = compute_l2_norm(env, vector);
  size_t len = env->vec_size(env, vector);

  double item;
  for (size_t i = 0; i < len; i++) {
    item = env->extract_float(env, env->vec_get(env, vector, i));
    item = item / fmax(norm, EPS);

    env->vec_set(env, vector, i, env->make_float(env, item));
  }
}

emacs_value Fnl2_normalize(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  nl2_normalize(env, args[0]);

  return env->intern(env, "nil");
}

int emacs_module_init(struct emacs_runtime* runtime) {
  if (runtime->size < sizeof(*runtime))
    return 1;

  emacs_env* env = runtime->get_environment(runtime);
  if (env->size < sizeof(*env))
    return 2;

  emacs_value nl2_normalize_fn = env->make_function(env, 1, 1, Fnl2_normalize, "Normalize numerical vector destructively.", NULL);
  emacs_value nl2_normalize_fn_args[] = {env->intern(env, "onnx-ml-utils-nl2-normalize"), nl2_normalize_fn};
  env->funcall(env, env->intern(env, "defalias"), 2, nl2_normalize_fn_args);

  emacs_value provide_fn_args[] = {env->intern(env, "onnx-ml-utils")};
  env->funcall(env, env->intern(env, "provide"), 1, provide_fn_args);

  return 0;
}
