package org.example;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;
import ai.djl.inference.Predictor;
import ai.djl.translate.Batchifier;

import java.io.IOException;
import java.nio.file.Paths;

// ===========================
// Custom Input Class
// ===========================
/**
 * Represents a single track input with 6 features (e.g., avgWire values).
 */
class TrackInput {
    float[] features;   // Length = 6

    /**
     * Constructor
     * @param features float array of input features (length 6)
     */
    public TrackInput(float[] features) {
        this.features = features;
    }
}

// ===========================
// Main Inference Program
// ===========================
/**
 * Main class for loading a TorchScript MLP model and performing single-sample inference.
 */
public class Main {

    public static void main(String[] args) {

        // -----------------------------
        // 1. Translator: TrackInput → Float (track probability)
        // -----------------------------
        Translator<TrackInput, Float> translator = new Translator<TrackInput, Float>() {

            @Override
            public NDList processInput(TranslatorContext ctx, TrackInput input) {
                NDManager manager = ctx.getNDManager();
                // Convert features to shape (1, 6)
                NDArray x = manager.create(input.features).reshape(1, input.features.length);
                return new NDList(x);
            }

            @Override
            public Float processOutput(TranslatorContext ctx, NDList list) {
                NDArray result = list.get(0);  // shape: (1,)
                return result.toFloatArray()[0];  // Extract single prediction
            }

            @Override
            public Batchifier getBatchifier() {
                return null;  // Single-sample inference, no batching
            }
        };

        // -----------------------------
        // 2. Define model loading criteria
        // -----------------------------
        Criteria<TrackInput, Float> criteria = Criteria.builder()
                .setTypes(TrackInput.class, Float.class)
                .optModelPath(Paths.get("nets/mlp_default.pt"))  // TorchScript model path
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        // -----------------------------
        // 3. Load model and run inference
        // -----------------------------
        try (ZooModel<TrackInput, Float> model = criteria.loadModel();
             Predictor<TrackInput, Float> predictor = model.newPredictor()) {

            // Example input with 6 float features
            float[] exampleFeatures = new float[]{44.6000f, 43.3333f, 41.0000f, 38.8571f, 35.1667f, 33.4286f};
            TrackInput input = new TrackInput(exampleFeatures);

            Float probability = predictor.predict(input);
            System.out.printf("Predicted track probability: %.4f%n", probability);

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException("Model inference failed", e);
        }
    }
}