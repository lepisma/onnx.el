;;; Unstructured interactive tests
;;; Code:

(progn
  (add-to-list 'load-path (file-name-directory (directory-file-name default-directory)))
  (require 'onnx)
  (require 'onnx-ml-utils)

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

;;; Tests for post-processing
(ert-deftest post-processing ()
  (let ((input [[1.0 1.0 1.0] [0.0 0.0 0.0]]))
    (onnx-ml-utils-nl2-normalize input)
    (should (equal input `[[,(/ 1.0 (sqrt 3.0)) ,(/ 1.0 (sqrt 3.0)) ,(/ 1.0 (sqrt 3.0))] [0.0 0.0 0.0]])))

  ;; Input is 2x4x5
  (let ((input [[[1.0 1.0 1.0 10.0 1.0] [1.0 1.0 2.0 20.0 1.0] [1.0 1.0 3.0 30.0 5.0] [1.0 1.0 4.0 40.0 50.0]]
	        [[0.0 0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0 0.0]]])
        (attention-mask [[1 1 1 0] [1 1 1 1]]))
    (should (equal (onnx-ml-utils-nmean-pool input attention-mask)
                   `[[1.0 1.0 2.0 20.0 ,(/ 7.0 3.0)] [0.0 0.0 0.0 0.0 0.0]]))))

;;; Tests for a simple single input single output model
(ert-deftest simple-model ()
  (let ((model (onnx-load "mean_20x100_to_20.onnx")))
    (should (equal (onnx-core-model-input-names model) '("input")))
    (should (equal (onnx-core-model-output-names model) '("output")))))

;; Manually run and test the values this
(let ((model (onnx-load "mean_20x100_to_20.onnx")))
  (onnx-run model `(("input" . ,(random-matrix '(20 100)))) '("output")))

;;; Tests for a complex sentence transformer model
;; This is the all-MiniLM-L6-v2 model's O2 version which you might
;; have to download (not shipped in the repository since it's a large
;; file)
(ert-deftest transformer-model ()
  (let ((model (onnx-load "../model_O2.onnx")))
    (should (equal (onnx-core-model-input-names model) '("input_ids" "attention_mask" "token_type_ids")))
    (should (equal (onnx-core-model-output-names model) '("last_hidden_state")))))

;; This is the test input, after tokenization, for the following list:
;; ["This is an example sentence" "Each sentence is converted"]
;; Run this manually and test
(let* ((model (onnx-load "../model_O2.onnx"))
       (input-ids [[101 2023 2003 2019 2742 6251 102]
                   [101 2169 6251 2003 4991 102  0  ]])
       (attention-mask [[1 1 1 1 1 1 1]
                        [1 1 1 1 1 1 0]])
       (token-type-ids [[0 0 0 0 0 0 0]
                        [0 0 0 0 0 0 0]])
       (output (onnx-run model `(("input_ids" . ,input-ids)
                                 ("attention_mask" . ,attention-mask)
                                 ("token_type_ids" . ,token-type-ids))
                         '("last_hidden_state"))))
  (setq output (onnx-ml-utils-nmean-pool output attention-mask))
  (onnx-ml-utils-nl2-normalize output)
  output)
