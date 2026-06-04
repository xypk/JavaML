package mlutils.java;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static mlutils.java.DataHandler.saveArray;
import static mlutils.java.DataHandler.setAttributes;

public class Experiment {
    public Model model;
    public Validation validator;
    public Evaluator eval;
    public DataSet train_set;
    public DataSet test_set;

    public Experiment(Model model, Validation validator, Evaluator eval, DataSet train_set, DataSet test_set) {
        this.model = model;
        this.validator = validator;
        this.eval = eval;
        this.train_set = train_set;
        this.test_set = test_set;
    }

    public static Experiment fromJson(String file_path) {
        JsonNode json_data = DataHandler.readJson(file_path);
        Experiment newExperiment = null;

        try {
            DataHandler DH = DataHandler.fromJson(json_data.get("dataset").asText());
            DataHandler[] train_and_test = DH.split(json_data.get("train_ratio").asDouble());
            DataHandler train_set = train_and_test[0];
            DataHandler test_set = train_and_test[1];
            int label_from = json_data.get("label_from").asInt();
            int label_to = json_data.get("label_to").asInt();

            Model model = Model.fromJson(json_data.get("model").asText());

            Evaluator eval = (Evaluator) Class.forName("mlutils.java.Evaluator$" + json_data.get("evaluator").get("type").asText()).getDeclaredConstructor().newInstance();
            Iterator<Map.Entry<String, JsonNode>> eval_params = json_data.get("evaluator").get("params").fields();
            setAttributes(eval, eval_params);

            Validation validator = (Validation) Class.forName("mlutils.java.Validation$" + json_data.get("validation").asText())
                    .getDeclaredConstructor(Model.class, RecordReaderDataSetIterator.class, Evaluator.class)
                    .newInstance(model, new RecordReaderDataSetIterator(train_set.Reader, train_set.getRows() / json_data.get("batches").asInt(), label_from, label_to, true), eval);

            JsonNode epochs = json_data.get("epochs");
            if (epochs != null) {
                model.setTrainParams(epochs.asInt());
            }

            newExperiment = new Experiment(
                    model,
                    validator,
                    eval,
                    (new RecordReaderDataSetIterator(train_set.Reader, train_set.getRows(), label_from, label_to, true)).next(),
                    (new RecordReaderDataSetIterator(test_set.Reader, test_set.getRows(), label_from, label_to, true)).next());
        }
        catch(ClassNotFoundException | NoSuchFieldException | NoSuchMethodException | InstantiationException |
              IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
        }

        return newExperiment;
    }

    public INDArray run(String pred_write_path, List<String> label_names) {
        validator.run();
        model.reset();
        model.fit(this.train_set.getFeatures(), this.train_set.getLabels());
        INDArray prediction = model.predict(this.test_set.getFeatures());
        if(pred_write_path != null) {
            if(label_names == null) {
                label_names = test_set.getLabelNamesList();
            }
            String generic_header = String.join(",", label_names);
            String final_header;
            if(label_names.size() > 1) {
                String pred_header = generic_header.replaceAll(",", " (Pred),");
                String truth_header = generic_header.replaceAll(",", " (Truth),");
                final_header = pred_header + "," + truth_header;
            } else {
                final_header = generic_header + " (Pred)," + generic_header + " (Truth)";
            }
            INDArray truth = this.test_set.getLabels();
            saveArray(pred_write_path, Nd4j.concat(1, prediction, truth), final_header);
            INDArray test_score = this.eval.evaluate(prediction, truth);
            System.out.printf("Test %s: ", eval.getName());
            System.out.print(test_score);
        }
        return prediction;
    }
}