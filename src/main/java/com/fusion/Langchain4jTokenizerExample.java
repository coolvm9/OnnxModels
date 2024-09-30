package com.fusion;

import dev.langchain4j.model.embedding.onnx.BertTokenizer;
import dev.langchain4j.model.embedding.onnx.HuggingFaceTokenizer;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Langchain4jTokenizerExample {

    public static void main(String[] args) {

        // Initialize the Hugging Face tokenizer from LangChain4j
        Path pathToTokenizer = Paths.get("/path/to/tokenizer.json");
        BertTokenizer tokenizer = new BertTokenizer(pathToTokenizer);
        // Input text for tokenization
        String text = "I love using Hugging Face models for NLP tasks.";

        // Tokenize the input text
        List<Integer> tokenizedIds = tokenizer.tokenize(text);

        // Print the tokenized IDs
        System.out.println("Tokenized IDs: " + tokenizedIds);

        // Decode the tokenized IDs back to text (optional)
        String decodedText = tokenizer.decode(tokenizedIds);
        System.out.println("Decoded text: " + decodedText);
    }
}