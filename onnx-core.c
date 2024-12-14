#include <emacs-module.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <onnxruntime_c_api.h>

int plugin_is_GPL_compatible;
const char* onnx_core_version = "0.0.1";

const OrtApi* g_ort = NULL;

// TODO: Use Emacs non local exit signal
#define ORT_ABORT_ON_ERROR(expr)                                \
  do {                                                          \
    OrtStatus* onnx_status = (expr);                            \
    if (onnx_status != NULL) {                                  \
      const char* msg = g_ort->GetErrorMessage(onnx_status);    \
      fprintf(stderr, ":: %s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                        \
      abort();                                                  \
    }                                                           \
  } while (0);


int run_inference(OrtSession* session, float* input, size_t input_size, float* output, size_t output_size) {
  int ret = 0;
  // TODO: Do model execution
  return ret;
}

// For now we will only work with single input and output models
void verify_input_output_count(OrtSession* session) {
  size_t count;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  assert(count == 1);
}

struct model_t {
  OrtEnv* env;
  OrtSessionOptions* session_options;
  OrtSession* session;
};

struct model_t load_model(ORTCHAR_T* model_path) {
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  /* if (!g_ort) { */
  /*   fprintf(stderr, "Failed to init ONNX Runtime engine.\n"); */
  /* } */

  ORTCHAR_T* execution_provider = "cpu";
  OrtEnv* env;

  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  assert(env != NULL);

  OrtSessionOptions* session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

  OrtSession* session;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

  // TODO: Enable check
  // verify_input_output_count(session);

  struct model_t model;
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
    // TODO
  }
  env->copy_string_contents(env, args[0], buf, &len);

  struct model_t* model = malloc(sizeof(struct model_t));
  if (!model) {
    // TODO
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

  emacs_env* env = runtime->get_environment(runtime);
  if (env->size < sizeof(*env))
    return 2;

  emacs_value version_fn = env->make_function(env, 0, 0, Fonnx_core_version, "Return version of onnx-core.", NULL);
  emacs_value version_fn_args[] = {env->intern(env, "onnx-core-version"), version_fn};
  env->funcall(env, env->intern(env, "defalias"), 2, version_fn_args);

  emacs_value load_fn = env->make_function(env, 1, 1, Fonnx_load_model, "Load given ONNX model file.", NULL);
  emacs_value load_fn_args[] = {env->intern(env, "onnx-core-load-model"), load_fn};
  env->funcall(env, env->intern(env, "defalias"), 2, load_fn_args);

  emacs_value provide_fn_args[] = {env->intern(env, "onnx-core")};
  env->funcall(env, env->intern(env, "provide"), 1, provide_fn_args);

  return 0;
}
