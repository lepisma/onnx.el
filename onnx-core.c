#include <emacs-module.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <onnxruntime_c_api.h>

int plugin_is_GPL_compatible;
const char* onnx_core_version = "0.0.3";

const OrtApi* g_ort = NULL;
emacs_env *g_env = NULL;

typedef enum emacs_vector_t {
  EMACS_VECTOR_INTEGER,
  EMACS_VECTOR_FLOAT,
  EMACS_VECTOR_UNKNOWN
} emacs_vector_t;

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

/*
 * Return nth item from an Emacs list. No sanity check is done here.
 */
emacs_value list_nth(emacs_env* env, emacs_value list, size_t n) {
  return env->funcall(env, env->intern(env, "nth"), 2, (emacs_value[]){env->make_integer(env, n), list});
}

/*
 * Take a single flat Emacs vector with floating values and convert to a C float
 * array.
 */
float* emacs_vector_to_float_array(emacs_env* env, emacs_value vector, size_t vector_len) {
  float* output = malloc(sizeof(float) * vector_len);

  for (size_t i = 0; i < vector_len; i++) {
    output[i] = env->extract_float(env, env->vec_get(env, vector, i));
  }

  return output;
}

/*
 * Take a single flat Emacs vector with integer values and convert to a C
 * int64_t array.
 */
int64_t* emacs_vector_to_int64_array(emacs_env* env, emacs_value vector, size_t vector_len) {
  int64_t* output = malloc(sizeof(int64_t) * vector_len);

  for (size_t i = 0; i < vector_len; i++) {
    output[i] = env->extract_integer(env, env->vec_get(env, vector, i));
  }

  return output;
}

/*
 * Return type of the Emacs vector by checking its first element. We assume
 * every other element will be of the same type. Not testing this might cause
 * issues at some point, but not today. Not today.
 */
emacs_vector_t emacs_vector_type(emacs_env* env, emacs_value vector) {
  // Assuming that the vector is not empty
  emacs_value it = env->vec_get(env, vector, 0);
  emacs_value type = env->funcall(env, env->intern(env, "type-of"), 1, (emacs_value[]){it});

  if (env->eq(env, type, env->intern(env, "float"))) {
    return EMACS_VECTOR_FLOAT;
  } else if (env->eq(env, type, env->intern(env, "integer"))) {
    return EMACS_VECTOR_INTEGER;
  } else {
    return EMACS_VECTOR_UNKNOWN;
  }
}

size_t prod_array_size_t(size_t* array, size_t len) {
  size_t output;
  for (size_t i = 0; i < len; i++) {
    if (i == 0) {
      output = array[i];
    } else {
      output = output * array[i];
    }
  }

  return output;
}

/*
 * Return a cons of vector representing the model output and the shape of the
 * vector. We assume that the caller will take care of length validity and other
 * basic checks.
 *
 * Arguments:
 * 0. model (user pointer)
 * 1. input-names (list of strings)
 * 2. output-names (list of strings)
 * 3. inputs (list of flat vectors)
 * 4. input-shapes (list of flat vector)
 */
emacs_value Fonnx_run(emacs_env* env, ptrdiff_t n, emacs_value args[], void* data) {
  struct model_t* model = env->get_user_ptr(env, args[0]);

  size_t n_inputs;
  char** input_names = get_list_of_strings(env, args[1], &n_inputs);

  size_t n_outputs;
  char** output_names = get_list_of_strings(env, args[2], &n_outputs);

  size_t input_lens[n_inputs];              // Lengths of input vectors
  void* input_vs[n_inputs];                 // Flat input vectors
  emacs_vector_t input_vs_types[n_inputs];  // Types of input vectors
  size_t input_shapes_lens[n_inputs];       // Lengths of input shape vectors
  size_t* input_shapes[n_inputs];           // Input shape vectors

  emacs_value vector;
  emacs_value shape_vector;
  for (size_t i = 0; i < n_inputs; i++) {
    vector = list_nth(env, args[3], i);
    input_vs_types[i] = emacs_vector_type(env, vector);
    shape_vector = list_nth(env, args[4], i);

    input_lens[i] = env->vec_size(env, vector);

    // Note that this type testing and parsing should ideally happen by checking
    // the model's inputs and not based on what the user is passing. This
    // difference might show up in places where there is a difference in number
    // of bits for the same broad type (say int with 8 bit or 64 bit).
    switch (input_vs_types[i]) {
    case EMACS_VECTOR_INTEGER:
      int64_t* int_input_vs = emacs_vector_to_int64_array(env, vector, input_lens[i]);
      input_vs[i] = (void*)int_input_vs;
      break;
    case EMACS_VECTOR_FLOAT:
      float* float_input_vs = emacs_vector_to_float_array(env, vector, input_lens[i]);
      input_vs[i] = (void*)float_input_vs;
      break;
    case EMACS_VECTOR_UNKNOWN:
      emacs_signal_error(env, "onnx-type-error", "Input vector of unknown type provided");
      return env->intern(env, "nil");
    default:
      emacs_signal_error(env, "onnx-type-error", "Input vector of unknown type provided");
      return env->intern(env, "nil");
    }

    input_shapes_lens[i] = env->vec_size(env, shape_vector);
    input_shapes[i] = emacs_vector_to_size_array(env, shape_vector, input_shapes_lens[i]);
  }

  OrtMemoryInfo* memory_info;
  ORT_RAISE_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

  OrtValue** input_tensors = malloc(sizeof(OrtValue*) * n_inputs);
  int is_tensor;
  for (size_t i = 0; i < n_inputs; i++) {
    // Since type check has happened earlier, we will just care about the two
    // valid branches here
    switch (input_vs_types[i]) {
    case EMACS_VECTOR_INTEGER:
      ORT_RAISE_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_vs[i], input_lens[i] * sizeof(int64_t),
                                                               input_shapes[i], input_shapes_lens[i],
                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[i]));
      break;
    case EMACS_VECTOR_FLOAT:
      ORT_RAISE_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_vs[i], input_lens[i] * sizeof(float),
                                                               input_shapes[i], input_shapes_lens[i],
                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[i]));
      break;
    }

    if (input_tensors[i] == NULL) {
        emacs_signal_error(env, "onnx-error", "One of the input tensors is NULL");
        return env->intern(env, "nil");
    }

    ORT_RAISE_ON_ERROR(g_ort->IsTensor(input_tensors[i], &is_tensor));
    if (!is_tensor) {
      emacs_signal_error(env, "onnx-error", "Unable to build input tensor");
      return env->intern(env, "nil");
    }
  }

  OrtValue* output_tensor = NULL;
  ORT_RAISE_ON_ERROR(g_ort->Run(model->session, NULL,
                                (const char* const*)input_names, (const OrtValue* const*)input_tensors, n_inputs,
                                (const char* const*)output_names, n_outputs, &output_tensor));
  g_ort->ReleaseMemoryInfo(memory_info);

  if (output_tensor == NULL) {
    emacs_signal_error(env, "onnx-error", "Output tensor is NULL");
  }
  ORT_RAISE_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
  if (!is_tensor) {
    emacs_signal_error(env, "onnx-error", "Unable to build output tensor");
  }

  // This is an assumption that the output type is known to be float
  float* output_tensor_data = NULL;
  ORT_RAISE_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));

  OrtTensorTypeAndShapeInfo* output_type_and_shape_info;
  ORT_RAISE_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensor, &output_type_and_shape_info));

  size_t output_shape_len;
  ORT_RAISE_ON_ERROR(g_ort->GetDimensionsCount(output_type_and_shape_info, &output_shape_len));

  int64_t* output_shape = malloc(sizeof(int64_t) * output_shape_len);
  ORT_RAISE_ON_ERROR(g_ort->GetDimensions(output_type_and_shape_info, output_shape, output_shape_len));

  size_t output_len = prod_array_size_t(output_shape, output_shape_len);

  emacs_value* emacs_output_items = malloc(sizeof(emacs_value) * output_len);
  for (size_t i = 0; i < output_len; i++) {
    emacs_output_items[i] = env->make_float(env, output_tensor_data[i]);
  }
  emacs_value emacs_output_vector = env->funcall(env, env->intern(env, "vector"), output_len, emacs_output_items);
  free(emacs_output_items);

  emacs_value* emacs_output_shape_items = malloc(sizeof(emacs_value) * output_shape_len);
  for (size_t i = 0; i < output_shape_len; i++) {
    emacs_output_shape_items[i] = env->make_integer(env, output_shape[i]);
  }
  emacs_value emacs_output_shape = env->funcall(env, env->intern(env, "vector"), output_shape_len, emacs_output_shape_items);
  free(emacs_output_shape_items);

  // Free all remaining input allocations
  for (size_t i = 0; i < n_inputs; i++) {
    g_ort->ReleaseValue(input_tensors[i]);
    free(input_vs[i]);
    free(input_shapes[i]);
  }
  free_char_array(input_names, n_inputs);

  // Free all remaining output allocations
  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseTensorTypeAndShapeInfo(output_type_and_shape_info);
  free_char_array(output_names, n_outputs);
  free(output_shape);

  return env->funcall(env, env->intern(env, "cons"), 2, (emacs_value[]){emacs_output_vector, emacs_output_shape});
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
