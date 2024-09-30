package com.fusion;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.onnxruntime.*;
import dev.langchain4j.model.embedding.onnx.HuggingFaceTokenizer;
import java.nio.file.Paths;
import java.util.*;

public class ONNXDistilBertExample {

    public static void main(String[] args) throws Exception {
        // Initialize the ONNX environment
        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {

            // Load ONNX model from file
            String modelPath = Paths.get("distilbert_sequence_classification.onnx").toAbsolutePath().toString();
            try (OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions())) {

                // Sample real input text
                String text = "I love using Hugging Face models for NLP tasks.";

                // Initialize the HuggingFaceTokenizer
                HuggingFaceTokenizer tokenizer = new HuggingFaceTokenizer(Paths.get("path/to/your/tokenizer.json"));

                // Tokenize the text
                long[][] inputIds = tokenizeText(tokenizer, text);

                // Convert to OnnxTensor
                OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputIds);

                // Run inference on the model
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", inputTensor);

                // Execute the session
                OrtSession.Result result = session.run(inputs);

                // Retrieve the output
                float[][] output = (float[][]) result.get(0).getValue();

                // Display the classification output (logits)
                System.out.println("Classification logits: " + Arrays.toString(output[0]));

                // Optionally, you can convert logits to the final class prediction
                int predictedClass = getPredictedClass(output[0]);
                System.out.println("Predicted Class: " + predictedClass);
            }
        }
    }

    // Tokenization using HuggingFaceTokenizer
    private static long[][] tokenizeText(HuggingFaceTokenizer tokenizer, String text) {
        // Tokenize the input text
        Encoding encoding = tokenizer.encode(text, false, true);
        long[] tokenIds = encoding.getIds();
        return new long[][] { tokenIds };
    }

    // Get predicted class from logits
    private static int getPredictedClass(float[] logits) {
        int predictedClass = 0;
        float maxLogit = logits[0];

        // Find the index with the highest logit value
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
                predictedClass = i;
            }
        }

        return predictedClass;
    }
}