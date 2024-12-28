;;; onnx.el --- ONNX Runtime binding -*- lexical-binding: t; -*-

;; Copyright (c) 2024 Abhinav Tushar

;; Author: Abhinav Tushar <abhinav@lepisma.xyz>
;; Version: 0.2.0
;; Package-Requires: ((emacs "29"))
;; Keywords: ml
;; URL: https://github.com/lepisma/onnx.el

;;; Commentary:

;; ONNX Runtime binding
;; This file is not a part of GNU Emacs.

;;; License:

;; This program is free software: you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program. If not, see <https://www.gnu.org/licenses/>.

;;; Code:

(require 'onnx-core)
(require 'cl-macs)
(require 'cl-extra)

(defun onnx-load (filepath)
  "Load onnx file from FILEPATH and return a model object."
  (let ((filepath (expand-file-name filepath)))
    (if (file-exists-p filepath)
        (onnx-core-load-model filepath)
      (error "Model file %s doesn't exist" filepath))))

(defun onnx-tokenize-text (text)
  "Tokenize TEXT and return vector that can be passed as input to a model.")

(defun onnx-input-names (model)
  "Return list of strings specifying input names for the MODEL."
  (onnx-core-model-input-names model))

(defun onnx-output-names (model)
  "Return list of strings specifying output names for the MODEL."
  (onnx-core-model-output-names model))

(defun onnx--matrix-shape (matrix &optional shape)
  "Return shape of a numeric MATRIX (based on Emacs vector) in form
of a list."
  (if (vectorp matrix)
      (onnx--matrix-shape (aref matrix 0) (cons (length matrix) shape))
    (reverse shape)))

(defun onnx--matrix-flatten (matrix)
  (if (vectorp (aref matrix 0))
      (apply #'vconcat (mapcar #'onnx--matrix-flatten matrix))
    matrix))

(defun onnx--vector-reshape (vector shape)
  "Convert a flat VECTOR to a matrix with given SHAPE (list)."
  (if (= 1 (length shape))
      vector
    (let* ((rows (car shape))
           (cols (apply #'* (cdr shape)))
           (step (/ (length vector) rows)))
      (apply #'vector
             (cl-loop for i from 0 below rows
                      collect (onnx--vector-reshape
                               (cl-subseq vector (* i step) (* (1+ i) step))
                               (cdr shape)))))))

(defun onnx-run (model input-alist output-names)
  "Run MODEL on INPUT-MATRIX and return the output vector.

INPUT-ALIST is an alist of string names mapping to shaped
matrices that will be taken up by the onnx runtime to pass on to
the model. OUTPUT-NAMES is a list of string names for output.

Input matrix should be of the right type (integer or float only
allowed right now) as required by the ONNX model."
  (if (= (length output-names) 1)
      (let ((result (onnx-core-run model (mapcar #'car input-alist) output-names
                                   (mapcar (lambda (it) (onnx--matrix-flatten (cdr it))) input-alist)
                                   (mapcar (lambda (it) (apply #'vector (onnx--matrix-shape (cdr it)))) input-alist))))
        (onnx--vector-reshape (car result) (mapcar #'identity (cdr result))))
    (error "Only single output is supported at the moment.")))

(provide 'onnx)

;;; onnx.el ends here
