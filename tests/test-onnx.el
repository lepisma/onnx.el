;;; Unstructured tests, can't be run automatically at the moment
;;; Code:

(progn
  (add-to-list 'load-path (file-name-directory (directory-file-name default-directory)))
  (require 'onnx)

  (defun random-vec (size)
    "Return a vector of SIZE with random floats between 0 to 1."
    (let ((res 100))
      (cl-loop repeat size
               vconcat (list (/ (random res)
				(float res))))))

  (defun random-matrix (shape)
    "Return a random matrix of the given SHAPE."
    (if (= (length shape) 1)
	(random-vec (car shape))
      (cl-loop repeat (car shape)
               vconcat (list (random-matrix (cdr shape)))))))

;;; Tests for basic loading and working via a mean pooling model
(setq model (onnx-load "mean_20x100_to_20.onnx"))
(onnx-core-model-input-names model)
;; -> ("input")
(onnx-core-model-output-names model)
;; -> ("output")
(onnx-run model `(("input" . ,(random-matrix '(20 100)))) '("output"))

;;; Tests for a complex sentence transformer model
;; This is the all-MiniLM-L6-v2 model's O2 version which you might
;; have to download (not shipped in the repository since it's a large
;; file)
(setq model (onnx-load "../model_O2.onnx"))
(onnx-core-model-input-names model)
;; -> ("input_ids" "attention_mask" "token_type_ids")
(onnx-core-model-output-names model)
;; -> ("last_hidden_state")

;; This is the test input, after tokenization, for the following python list:
;; sentences = ["This is an example sentence", "Each sentence is converted"]
(let ((input-ids [[101 2023 2003 2019 2742 6251 102]
                  [101 2169 6251 2003 4991 102  0  ]])
      (attention-mask [[1 1 1 1 1 1 1]
                       [1 1 1 1 1 1 0]])
      (token-type-ids [[0 0 0 0 0 0 0]
                       [0 0 0 0 0 0 0]]))
  (onnx-run model `(("input_ids" . ,input-ids)
                    ("attention_mask" . ,attention-mask)
                    ("token_type_ids" . ,token-type-ids))
            '("last_hidden_state")))

;; This test uses the tokenizers.el library to go from text to embedding
(require 'tokenizers)

(setq tk (tokenizers-from-pretrained "sentence-transformers/all-MiniLM-L6-v2"))
;; This is important for getting the same results as in
;; sentence-transformers
(tokenizers-enable-padding tk 0 "[PAD]")

(let* ((sentences ["This is an example sentence" "Each sentence is converted"])
       (tk-output (tokenizers-encode-batch tk sentences t)))
  (onnx-run model `(("input_ids" . ,(nth 0 tk-output))
                    ("attention_mask" . ,(nth 2 tk-output))
                    ("token_type_ids" . ,(nth 1 tk-output)))
            '("last_hidden_state")))

;;; Tests for post-processing
(require 'onnx-ml-utils)

(let ((input [[1.0 1.0 1.0] [0.0 0.0 0.0]]))
  (onnx-ml-utils-nl2-normalize input)
  input) ; (/ 1 (sqrt 3))

;; Input is 2x4x5
(let ((input [[[1.0 1.0 1.0 10.0 1.0] [1.0 1.0 2.0 20.0 1.0] [1.0 1.0 3.0 30.0 5.0] [1.0 1.0 4.0 40.0 50.0]]
	      [[0.0 0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0 0.0]]])
      (attention-mask [[1 1 1 0] [1 1 1 1]]))
  (onnx-ml-utils-nmean-pool input attention-mask))
