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

void clean_up_model(struct model_t model) {
  g_ort->ReleaseSessionOptions(model.session_options);
  g_ort->ReleaseSession(model.session);
  g_ort->ReleaseEnv(model.env);
}

void fin_model(void* ptr) {
  struct model_t* model = (struct model_t*)ptr;
  if (model) {
    clean_up_model(*model);
    free(model);
  }
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

  emacs_value provide_fn_args[] = {g_env->intern(g_env, "onnx-core")};
  g_env->funcall(g_env, g_env->intern(g_env, "provide"), 1, provide_fn_args);

  return 0;
}
