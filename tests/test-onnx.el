;;; Unstructured tests, can't be run automatically at the moment
;;; Code:

(progn
  (add-to-list 'load-path (file-name-directory (directory-file-name default-directory)))
  (require 'onnx)

  (setq model (onnx-load "mean_20x100_to_20.onnx"))

  (onnx-core-model-input-names model)
  ;; -> ("input")

  (onnx-core-model-output-names model)
  ;; -> ("output")

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

(onnx-run model `(("input" . ,(random-matrix '(20 100)))) '("output"))

