#include <emacs-module.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <onnxruntime_c_api.h>

int plugin_is_GPL_compatible;
const char* onnx_core_version = "0.0.1";

const OrtApi* g_ort = NULL;
emacs_env *g_env = NULL;

void emacs_signal_error(emacs_env *env, const char* code, const char* message) {
  emacs_value err_sym = env->intern(env, code);
  emacs_value data = env->make_string(env, message, strlen(message));
  env->non_local_exit_signal(env, err_sym, data);
}

#define ORT_RAISE_ON_ERROR(expr)                                \
  do {                                                          \
    OrtStatus* onnx_status = (expr);                            \
    if (onnx_status != NULL) {                                  \
      const char* msg = g_ort->GetErrorMessage(onnx_status);    \
      emacs_signal_error(g_env, "onnx-error", msg);             \
      g_ort->ReleaseStatus(onnx_status);                        \
    }                                                           \
  } while (0);

void run_inference(OrtSession* session, float* input, size_t input_size, float* output, size_t output_size) {
}

// For now we only work with single input and output models
void verify_input_output_count(OrtSession* session) {
  size_t count;
  ORT_RAISE_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  if (count != 1) {
    emacs_signal_error(g_env, "onnx-model-error", "Model input count is not 1");
  }
  ORT_RAISE_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  if (count != 1) {
    emacs_signal_error(g_env, "onnx-model-error", "Model output count is not 1");
  }
}

struct model_t {
  OrtEnv* env;
  OrtSessionOptions* session_options;
  OrtSession* session;
};

struct model_t load_model(ORTCHAR_T* model_path) {
  struct model_t model;

  if (!g_ort) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
      emacs_signal_error(g_env, "onnx-error", "Failed to initialize ORT API");
      return model;
    }
  }

  ORTCHAR_T* execution_provider = "cpu";
  OrtEnv* env;

  ORT_RAISE_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  if (env == NULL) {
    emacs_signal_error(g_env, "onnx-error", "Failed to create environment");
    return model;
  }

  OrtSessionOptions* session_options;
  ORT_RAISE_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

  OrtSession* session;
  ORT_RAISE_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

  // TODO: Enable check
  // verify_input_output_count(session);

  model.env = env;
  model.session_options = session_options;
  model.session = session;

  return model;
}

size_t model_input_count(struct model_t* model) {
  size_t n;
  ORT_RAISE_ON_ERROR(g_ort->SessionGetInputCount(model->session, &n));
  return n;
}

char** model_input_names(struct model_t* model) {
  size_t n = model_input_count(model);

  OrtAllocator* allocator;
  ORT_RAISE_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  char** input_names = malloc(sizeof(char*) * n);

  size_t i;
  for (i = 0; i < n; i++) {
    char* name;
    ORT_RAISE_ON_ERROR(g_ort->SessionGetInputName(model->session, i, allocator, &name));
    input_names[i] = malloc(strlen(name) + 1);
    strcpy(input_names[i], name);
    ORT_RAISE_ON_ERROR(g_ort->AllocatorFree(allocator, name));
  }

  return input_names;
}

size_t model_output_count(struct model_t* model) {
  size_t n;
  ORT_RAISE_ON_ERROR(g_ort->SessionGetOutputCount(model->session, &n));
  return n;
}

char** model_output_names(struct model_t* model) {
  size_t n = model_output_count(model);

  OrtAllocator* allocator;
  ORT_RAISE_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  char** output_names = malloc(sizeof(char*) * n);

  size_t i;
  for (i = 0; i < n; i++) {
    char* name;
    ORT_RAISE_ON_ERROR(g_ort->SessionGetOutputName(model->session, i, allocator, &name));
    output_names[i] = malloc(strlen(name) + 1);
    strcpy(output_names[i], name);
    ORT_RAISE_ON_ERROR(g_ort->AllocatorFree(allocator, name));
  }

  return output_names;
}

void clean_up_model(struct model_t* model) {
  g_ort->ReleaseSessionOptions(model->session_options);
  g_ort->ReleaseSession(model->session);
  g_ort->ReleaseEnv(model->env);
}

void fin_model(void* ptr) {
  struct model_t* model = (struct model_t*)ptr;
  if (model) {
    clean_up_model(model);
    free(model);
  }
}

void free_char_array(char** array, size_t count) {
    if (!array) {
        return;
    }

    for (size_t i = 0; i < count; i++) {
        if (array[i]) {
            free(array[i]);
        }
    }

    free(array);
}

emacs_value Fonnx_load_model(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  ptrdiff_t len;
  env->copy_string_contents(env, args[0], NULL, &len);
  char* buf = malloc(len);
  if (!buf) {
    emacs_signal_error(g_env, "onnx-memory-error", "Failed to allocate for filepath string");
    return NULL;
  }
  env->copy_string_contents(env, args[0], buf, &len);

  struct model_t* model = malloc(sizeof(struct model_t));
  if (!model) {
    emacs_signal_error(g_env, "onnx-memory-error", "Failed to allocate for model");
    return NULL;
  }
  *model = load_model(buf);
  free(buf);

  return env->make_user_ptr(env, fin_model, model);
}

emacs_value Fonnx_model_input_names(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  struct model_t* model = env->get_user_ptr(env, args[0]);

  char** input_names = model_input_names(model);
  size_t input_count = model_input_count(model);

  size_t i;
  emacs_value* emacs_strings = malloc(sizeof(emacs_value) * input_count);

  for (i = 0; i < input_count; i++) {
    emacs_strings[i] = env->make_string(env, input_names[i], strlen(input_names[i]));
  }

  emacs_value result = env->funcall(env, env->intern(env, "list"), input_count, emacs_strings);

  free(emacs_strings);
  free_char_array(input_names, input_count);

  return result;
}

emacs_value Fonnx_model_output_names(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  struct model_t* model = env->get_user_ptr(env, args[0]);

  char** output_names = model_output_names(model);
  size_t output_count = model_output_count(model);

  size_t i;
  emacs_value* emacs_strings = malloc(sizeof(emacs_value) * output_count);

  for (i = 0; i < output_count; i++) {
    emacs_strings[i] = env->make_string(env, output_names[i], strlen(output_names[i]));
  }

  emacs_value result = env->funcall(env, env->intern(env, "list"), output_count, emacs_strings);

  free(emacs_strings);
  free_char_array(output_names, output_count);

  return result;
}

/*
 * Read and return a list of strings from ARG. Set LIST_LEN to be the length of
 * the list.
 */
char** get_list_of_strings(emacs_env* env, emacs_value list, size_t* list_len) {
  emacs_value e_len = env->funcall(env, env->intern(env, "length"), 1, (emacs_value[]){list});
  *list_len = env->extract_integer(env, e_len);

  char** output = malloc(sizeof(char*) * (*list_len));

  emacs_value item;
  size_t str_len;
  for (size_t i = 0; i < *list_len; i++) {
    item = env->funcall(env, env->intern(env, "nth"), 2, (emacs_value[]){env->make_integer(env, i), list});
    env->copy_string_contents(env, item, NULL, &str_len);
    char* str = malloc(sizeof(char) * str_len);
    env->copy_string_contents(env, item, str, &str_len);
    output[i] = str;
  }

  return output;
}

size_t* emacs_vector_to_size_array(emacs_env* env, emacs_value vector, size_t vector_len) {
  size_t* output = malloc(sizeof(size_t) * vector_len);

  for (size_t i = 0; i < vector_len; i++) {
    output[i] = env->extract_integer(env, env->vec_get(env, vector, i));
  }

  return output;
}

float* emacs_vector_to_float_array(emacs_env* env, emacs_value vector, size_t vector_len) {
  float* output = malloc(sizeof(float) * vector_len);

  for (size_t i = 0; i < vector_len; i++) {
    output[i] = env->extract_float(env, env->vec_get(env, vector, i));
  }

  return output;
}

/*
 * Return an Emacs vector representing the model output
 *
 * Arguments:
 * 0. model (user pointer)
 * 1. input-names (list)
 * 2. output-names (list)
 * 3. input (flat vector)
 * 4. input-shape (flat vector)
 */
emacs_value Fonnx_run(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  struct model_t* model = env->get_user_ptr(env, args[0]);
}

static emacs_value Fonnx_core_version(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  return env->make_string(env, onnx_core_version, strlen(onnx_core_version));
}

int emacs_module_init(struct emacs_runtime* runtime) {
  if (runtime->size < sizeof(*runtime))
    return 1;

  g_env = runtime->get_environment(runtime);
  if (g_env->size < sizeof(*g_env))
    return 2;

  emacs_value version_fn = g_env->make_function(g_env, 0, 0, Fonnx_core_version, "Return version of onnx-core.", NULL);
  emacs_value version_fn_args[] = {g_env->intern(g_env, "onnx-core-version"), version_fn};
  g_env->funcall(g_env, g_env->intern(g_env, "defalias"), 2, version_fn_args);

  emacs_value load_fn = g_env->make_function(g_env, 1, 1, Fonnx_load_model, "Load given ONNX model file.", NULL);
  emacs_value load_fn_args[] = {g_env->intern(g_env, "onnx-core-load-model"), load_fn};
  g_env->funcall(g_env, g_env->intern(g_env, "defalias"), 2, load_fn_args);

  emacs_value input_names_fn = g_env->make_function(g_env, 1, 1, Fonnx_model_input_names, "Return a list of input names for the model.", NULL);
  emacs_value input_names_fn_args[] = {g_env->intern(g_env, "onnx-core-model-input-names"), input_names_fn};
  g_env->funcall(g_env, g_env->intern(g_env, "defalias"), 2, input_names_fn_args);

  emacs_value output_names_fn = g_env->make_function(g_env, 1, 1, Fonnx_model_output_names, "Return a list of output names for the model.", NULL);
  emacs_value output_names_fn_args[] = {g_env->intern(g_env, "onnx-core-model-output-names"), output_names_fn};
  g_env->funcall(g_env, g_env->intern(g_env, "defalias"), 2, output_names_fn_args);

  emacs_value run_fn = g_env->make_function(g_env, 5, 5, Fonnx_run, "Run given vector via the model.", NULL);
  emacs_value run_fn_args[] = {g_env->intern(g_env, "onnx-core-run"), run_fn};
  g_env->funcall(g_env, g_env->intern(g_env, "defalias"), 2, run_fn_args);

  emacs_value provide_fn_args[] = {g_env->intern(g_env, "onnx-core")};
  g_env->funcall(g_env, g_env->intern(g_env, "provide"), 1, provide_fn_args);

  return 0;
}
