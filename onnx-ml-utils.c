#include <emacs-module.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
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

/*
 * L2 normalize a 2D matrix along axis=1
 */
emacs_value Fnl2_normalize(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  size_t n_rows = env->vec_size(env, args[0]);

  for (size_t i = 0; i < n_rows; i++) {
    nl2_normalize(env, env->vec_get(env, args[0], i));
  }

  return env->intern(env, "nil");
}

/*
 * Destructively mask a float matrix (Batch x Length x Embedding dimension)
 * vector using integer mask matrix (Batch x Length).
 */
void nmask(emacs_env* env, emacs_value matrix, emacs_value mask_matrix) {
  size_t n_batch = env->vec_size(env, matrix);
  size_t n_length = env->vec_size(env, env->vec_get(env, matrix, 0));
  size_t n_dim = env->vec_size(env, env->vec_get(env, env->vec_get(env, matrix, 0), 0));

  int mask;
  for (size_t i = 0; i < n_batch; i++) {
    for (size_t j = 0; j < n_length; j++) {
      mask = env->extract_integer(env, env->vec_get(env, env->vec_get(env, mask_matrix, i), j));
      for (size_t k = 0; k < n_dim; k++) {
        if (mask == 0) {
          env->vec_set(env, env->vec_get(env, env->vec_get(env, matrix, i), j), k, env->make_float(env, 0.0));
        }
      }
    }
  }
}

emacs_value Fnmean_pool(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  nmask(env, args[0], args[1]);

  // All these assume that we really get 3D matrix as input
  size_t n_batch = env->vec_size(env, args[0]);
  size_t n_length = env->vec_size(env, env->vec_get(env, args[0], 0));
  size_t n_dim = env->vec_size(env, env->vec_get(env, env->vec_get(env, args[0], 0), 0));

  // Count (not really since this also has an EPS in case the value is zero) of
  // non-zero elements in each item of the batch. This will become the
  // denominator in mean division.
  double* elem_counts = malloc(sizeof(double) * n_batch);
  emacs_value vec;
  for (size_t i = 0; i < n_batch; i++) {
    vec = env->vec_get(env, args[1], i);

    elem_counts[i] = 0;
    for (size_t j = 0; j < n_length; j++) {
      elem_counts[i] = elem_counts[i] + env->extract_integer(env, env->vec_get(env, vec, j));
    }
    if (elem_counts[i] == 0) {
      elem_counts[i] = EPS;
    }
  }

  // Sums along the n_length dimension
  emacs_value output = env->funcall(env, env->intern(env, "make-vector"), 2, (emacs_value[]){env->make_integer(env, n_batch), env->make_integer(env, 0)});
  for (size_t i = 0; i < n_batch; i++) {
    env->vec_set(env, output, i, env->funcall(env, env->intern(env, "make-vector"), 2, (emacs_value[]){env->make_integer(env, n_dim), env->make_integer(env, 0)}));
    for (size_t k = 0; k < n_dim; k++) {
      double sum = 0;
      for (size_t j = 0; j < n_length; j++) {
        sum += env->extract_float(env, env->vec_get(env, env->vec_get(env, env->vec_get(env, args[0], i), j), k));
      }
      env->vec_set(env, env->vec_get(env, output, i), k, env->make_float(env, sum / elem_counts[i]));
    }
  }

  free(elem_counts);
  return output;
}

int emacs_module_init(struct emacs_runtime* runtime) {
  if (runtime->size < sizeof(*runtime))
    return 1;

  emacs_env* env = runtime->get_environment(runtime);
  if (env->size < sizeof(*env))
    return 2;

  emacs_value nl2_normalize_fn = env->make_function(env, 1, 1, Fnl2_normalize, "Normalize numerical 2D matrix (Batch x Embedding dimension) destructively.", NULL);
  emacs_value nl2_normalize_fn_args[] = {env->intern(env, "onnx-ml-utils-nl2-normalize"), nl2_normalize_fn};
  env->funcall(env, env->intern(env, "defalias"), 2, nl2_normalize_fn_args);

  emacs_value nmean_pool_fn = env->make_function(env, 2, 2, Fnmean_pool, "Mean pool input matrix (Batch x Length x Embedding dimension) with attention mask (Batch x Length) and return new matrix (Batch x Embedding Dimension), destructively.", NULL);
  emacs_value nmean_pool_fn_args[] = {env->intern(env, "onnx-ml-utils-nmean-pool"), nmean_pool_fn};
  env->funcall(env, env->intern(env, "defalias"), 2, nmean_pool_fn_args);

  emacs_value provide_fn_args[] = {env->intern(env, "onnx-ml-utils")};
  env->funcall(env, env->intern(env, "provide"), 1, provide_fn_args);

  return 0;
}
