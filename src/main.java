import mlutils.java.*;

void main() {
    // Example: Predict when a machine is going to fail with Gaussian Naive Bayes
    Experiment.fromJson("example_json/CustomExperiment.json").run("example_out/machine_pred.csv", List.of("fail"));
    System.exit(0);
}