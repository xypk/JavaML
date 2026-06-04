package mlutils.java;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.databind.JsonNode;

import javax.swing.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;
import java.util.Map;

import static java.lang.Math.PI;
import static java.lang.Math.sqrt;
import static mlutils.java.DataHandler.setAttributes;
import static org.nd4j.linalg.factory.Nd4j.ones;
import static org.nd4j.linalg.factory.Nd4j.zeros;

public abstract class Model {
    protected String Name;

    protected final int x_size;
    protected final int y_size;
    protected final Filter filter;
    protected INDArray out;

    protected final JFrame frame = new JFrame();
    protected final JPanel panel = new JPanel();

    public Model(int x_size, int y_size, Filter filter) {
        this.x_size = x_size;
        this.y_size = y_size;
        this.filter = filter;

        this.frame.add(this.panel);
        this.frame.setSize(500,100);
        this.panel.setVisible(true);
    }

    abstract void fit(INDArray input, INDArray output);
    abstract void once(INDArray input);

    public void reset() {}

    public void setTrainParams(int epochs) {}

    public INDArray predict(INDArray input) {
        JProgressBar pbar = new JProgressBar(0, 100);
        pbar.setStringPainted(true);
        pbar.setValue(0);
        this.panel.add(pbar);
        this.frame.setTitle(String.format("%s Prediction Status", this.Name));
        this.frame.setVisible(true);

        INDArray pred = Nd4j.create(new float[input.rows()][this.y_size]);
        for(int i = 0; i < input.rows(); i++) {
            once(input.getRow(i));
            pred.putRow(i, this.out);
            pbar.setValue((int) (100 * (double) i / (input.rows()-1)));
        }

        pbar.setValue(100);
        this.frame.setVisible(false);
        this.panel.remove(pbar);
        return this.filter.out(pred);
    }

    public static Model fromJson(String file_path) {
        JsonNode json_data = DataHandler.readJson(file_path);
        Model newModel = null;

        try {
            int x_size = json_data.get("x_size").asInt();
            int y_size = json_data.get("y_size").asInt();
            Filter filter = (Filter) Class.forName("mlutils.java.Filter$" + json_data.get("filter").get("type").asText()).getDeclaredConstructor().newInstance();
            Iterator<Map.Entry<String, JsonNode>> filter_params = json_data.get("filter").get("params").fields();
            setAttributes(filter, filter_params);

            switch (json_data.get("type").asText()) {
                case "NN":
                    NeuralNetConfiguration.ListBuilder LB = new NeuralNetConfiguration.Builder()
                            .seed(json_data.get("seed").asInt())
                            .updater((IUpdater) Class.forName("org.nd4j.linalg.learning.config." + json_data.get("updater").get("type").asText()).getDeclaredConstructor(double.class).newInstance(json_data.get("updater").get("learning_rate").asDouble()))
                            .list();
                    Iterator layers = json_data.get("layers").elements();
                    while (layers.hasNext()) {
                        JsonNode layer = (JsonNode) layers.next();
                        Layer LC = null;
                        switch (layer.get("type").asText()) {
                            case "dense":
                                DenseLayer.Builder LCB_Dense = new DenseLayer.Builder();
                                LCB_Dense = LCB_Dense.nIn(layer.get("in").asInt());
                                LCB_Dense = LCB_Dense.nOut(layer.get("out").asInt());
                                JsonNode activation = layer.get("activation");
                                if(activation != null) {
                                    LCB_Dense.activation(Activation.fromString(activation.asText()));
                                }
                                LC = LCB_Dense.build();
                                break;
                            case "output":
                                OutputLayer.Builder LCB_Output = new OutputLayer.Builder();
                                LCB_Output = LCB_Output.nIn(layer.get("in").asInt());
                                LCB_Output = LCB_Output.nOut(layer.get("out").asInt());

                                JsonNode activationName = layer.get("activation");
                                if (activationName != null && !activationName.isNull()) {
                                    LCB_Output = LCB_Output.activation(Activation.fromString(activationName.asText()));
                                }

                                JsonNode lossName = layer.get("loss_fn");
                                if (lossName != null && !lossName.isNull()) {
                                    LCB_Output = LCB_Output.lossFunction(LossFunctions.LossFunction.valueOf(lossName.asText()));
                                }

                                LC = LCB_Output.build();
                                break;
                        }
                        LB = LB.layer(layer.get("index").asInt(), LC);
                    }
                    MultiLayerConfiguration MLC = LB.build();
                    NN newNN = new NN(MLC, x_size, y_size, filter);
                    newNN.init();
                    newModel = newNN;
                    break;
                case "KernelDensityNB":
                    Kernel kernel = (Kernel) Class.forName("mlutils.java.Kernel$" + json_data.get("kernel").asText()).getDeclaredConstructor().newInstance();
                    int train_grid_n = json_data.get("train_grid").asInt();
                    int out_grid_n = json_data.get("out_grid").asInt();
                    Evaluator eval = (Evaluator) Class.forName("mlutils.java.Evaluator$" + json_data.get("loss_fn").get("type").asText()).getDeclaredConstructor().newInstance();
                    Iterator<Map.Entry<String, JsonNode>> eval_params = json_data.get("loss_fn").get("params").fields();
                    setAttributes(eval, eval_params);
                    newModel = new KernelDensityNB(x_size, y_size, kernel, filter, train_grid_n, out_grid_n, eval);
                    break;
                default:
                    newModel = (Model) Class.forName("mlutils.java.Model$" + json_data.get("type").asText()).getDeclaredConstructor(int.class, int.class, Filter.class).newInstance(x_size, y_size, filter);
                    break;
            }
        }
        catch(ClassNotFoundException | InvocationTargetException | NoSuchMethodException | NoSuchFieldException | InstantiationException | IllegalAccessException e) {
            e.printStackTrace();
        }

        return newModel;
    }

    public static class NN extends Model {
        private final MultiLayerConfiguration config;
        private MultiLayerNetwork network;

        private long epochs = 1;
        private INDArray reset_params = null;

        public NN(MultiLayerConfiguration config, int x_size, int y_size, Filter filter) {
            super(x_size, y_size, filter);
            this.config = config;
            this.network = new MultiLayerNetwork(config);
            this.Name = "Neural Network";
        }

        public void init() {
            this.network.init();
            this.setResetParams(this.getParams().dup());
        }
        public void init(INDArray parameters, boolean cloneParametersArray) {
            this.network.init(parameters, cloneParametersArray);
            this.setResetParams(this.getParams().dup());
        }

        @Override
        public void setTrainParams(int epochs) {
            this.epochs = epochs;
        }

        public INDArray getParams() {
            return this.network.params();
        }

        public void setResetParams(INDArray params) {
            this.reset_params = params;
        }

        @Override
        public void reset() {
            this.network = new MultiLayerNetwork(config);
            this.network.init();
            this.network.setParameters(this.reset_params.dup().reshape(1,-1));
        }

        public void fit(INDArray input, INDArray output) {
            JProgressBar pbar = new JProgressBar(0, 100);
            pbar.setStringPainted(true);
            pbar.setValue(0);
            this.panel.add(pbar);
            this.frame.setTitle(String.format("%s Training Status", this.Name));
            this.frame.setVisible(true);

            for(int i = 0; i < this.epochs; i++) {
                // One epoch
                this.network.fit(input, output);

                // Calculate loss
                double loss = this.network.score();
                //System.out.println(loss);

                // Progress bar
                pbar.setValue((int) (100 * (double) i / (this.epochs-1)));
            }

            pbar.setValue(100);
            this.frame.setVisible(false);
            this.panel.remove(pbar);
        }

        public void once(INDArray input) {
            this.out = this.network.output(input);
        }

        @Override
        public INDArray predict(INDArray input) {
            this.once(input);
            return this.filter.out(this.out);
        }

        public void saveWeights(String path) throws IOException {
            File file = new File(path);
            this.network.save(file);
        }

        public void loadWeights(String path) throws IOException {
            File file = new File(path);
            this.network = MultiLayerNetwork.load(file, false);
        }
    }

    public static class GaussianNB extends DiscreteNB {
        private static final double sqrt2pi = sqrt(2*PI);

        private INDArray mu_x_y1; // mean of x given y=1
        private INDArray sigma_x_y1; // stdev of x given y=1
        private INDArray mu_x_y0; // mean of x given y=0
        private INDArray sigma_x_y0; // stdev of x given y=0

        private static INDArray gaussianDist(INDArray x, INDArray mu, INDArray sigma) {
            return Transforms.exp(Transforms.pow(x.sub(mu).div(sigma),2).div(-2)).div(sigma.mul(sqrt2pi));
        }

        public GaussianNB(int x_size, int y_size, Filter filter) {
            super(x_size, y_size, filter);
            this.mu_x_y1 = Nd4j.create(new double[y_size][x_size]);
            this.sigma_x_y1 = Nd4j.create(new double[y_size][x_size]);
            this.mu_x_y0 = Nd4j.create(new double[y_size][x_size]);
            this.sigma_x_y0 = Nd4j.create(new double[y_size][x_size]);
            this.Name = "Gaussian NB";
        }

        public void fit(INDArray input, INDArray labels) {
            calcYProb(labels);

            for(int i = 0; i < labels.size(1); i++) {
                INDArray indices = labels.getColumn(i).eq(1);
                if(indices.any()) {
                    INDArray x_y1 = input.get(Nd4j.where(indices, null, null)[0]); // x given y=1
                    this.mu_x_y1.putRow(i, x_y1.mean(0));
                    this.sigma_x_y1.putRow(i, x_y1.std(0));
                } else {
                    this.mu_x_y1.putRow(i, zeros(this.x_size));
                    this.sigma_x_y1.putRow(i, zeros(this.x_size));
                }
            }

            for(int i = 0; i < labels.size(1); i++) {
                INDArray indices = labels.getColumn(i).eq(0);
                if(indices.any()) {
                    INDArray x_y0 = input.get(Nd4j.where(indices, null, null)[0]); // x given y=0
                    this.mu_x_y0.putRow(i, x_y0.mean(0));
                    this.sigma_x_y0.putRow(i, x_y0.std(0));
                } else {
                    this.mu_x_y0.putRow(i, zeros(this.x_size));
                    this.sigma_x_y0.putRow(i, zeros(this.x_size));
                }
            }
        }

        void once(INDArray input) {
            INDArray repeated_input = ones(this.y_size,1).castTo(DataType.FLOAT).mmul(input.reshape(1,this.x_size));
            this.x_y1_prod = gaussianDist(input, this.mu_x_y1, this.sigma_x_y1).prod(1).mul(this.prob_y1);
            this.x_y0_prod = gaussianDist(repeated_input, this.mu_x_y0, this.sigma_x_y0).prod(1).mul(this.prob_y1.sub(1).neg());
            this.out = getRelativeProb();
        }
    }

    abstract static class DiscreteNB extends Model {
        protected INDArray x_y1_prod; // product of probability of input given y=1
        protected INDArray x_y0_prod; // product of probability of input given y=0
        protected INDArray prob_y1; // probability of y=1

        protected DiscreteNB(int x_size, int y_size, Filter filter) {
            super(x_size, y_size, filter);
            this.x_y1_prod = Nd4j.create(new double[y_size]);
            this.x_y0_prod = Nd4j.create(new double[y_size]);
            this.prob_y1 = Nd4j.create(new double[y_size]);
            this.Name = "Abstract Discrete NB";
        }

        protected void calcYProb(INDArray labels) {
            this.prob_y1 = labels.castTo(DataType.FLOAT).sum(0).div(labels.size(0));
        }

        protected INDArray getRelativeProb() {
            return this.x_y1_prod.div(this.x_y1_prod.add(this.x_y0_prod));
        }

        public static class NominalDiscreteNB extends DiscreteNB {
            private INDArray prob_x1_y1; // probability of x=1 given y=1
            private INDArray prob_x1_y0; // probability of x=1 given y=0

            public double epsilon = Float.MIN_VALUE; // small value added to zeros in product to prevent collapsing

            public NominalDiscreteNB(int x_size, int y_size, Filter filter) {
                super(x_size, y_size, filter);
                this.prob_x1_y1 = Nd4j.create(new double[y_size][x_size]);
                this.prob_x1_y0 = Nd4j.create(new double[y_size][x_size]);
                this.Name = "Nominal Discrete NB";
            }

            public void fit(INDArray input, INDArray labels) {
                calcYProb(labels);

                for(int i = 0; i < labels.size(1); i++) {
                    INDArray x_y1 = input.get(Nd4j.where(labels.getColumn(i).eq(1),null,null)[0]); // x given y=1
                    this.prob_x1_y1.putRow(i, x_y1.castTo(DataType.DOUBLE).mean(0));
                }

                for(int i = 0; i < labels.size(1); i++) {
                    INDArray x_y0 = input.get(Nd4j.where(labels.getColumn(i).eq(0), null, null)[0]); // x given y=0
                    this.prob_x1_y0.putRow(i, x_y0.castTo(DataType.DOUBLE).mean(0));
                }
            }

            void once(INDArray input) {
                this.x_y1_prod = this.prob_x1_y1.mul(input.eq(1)).add(this.epsilon).prod(1).mul(this.prob_x1_y1.mul(input.eq(0)).sub(1).neg().add(this.epsilon).prod(1)).mul(this.prob_y1);
                this.x_y0_prod = this.prob_x1_y0.mul(input.eq(1)).add(this.epsilon).prod(1).mul(this.prob_x1_y0.mul(input.eq(0)).sub(1).neg().add(this.epsilon).prod(1)).mul(this.prob_y1.sub(1).neg());
                this.out = getRelativeProb();
            }
        }
    }

    public static class KernelDensityNB extends Model {
        private static final double epsilon = 0.001;
        private final Kernel kernel;
        private INDArray c_x;
        private INDArray c_y;
        private INDArray h_x;
        private INDArray h_y;
        private int n;
        private INDArray x_examples;
        private INDArray y_examples;
        private final int train_grid_n;
        private final int out_grid_n;
        private final Evaluator loss_fn;

        public KernelDensityNB(int x_size, int y_size, Kernel kernel, Filter filter, int train_grid_n, int out_grid_n, Evaluator loss_fn) {
            super(x_size, y_size, filter);
            this.h_x = Nd4j.create(new double[x_size]);
            this.h_y = Nd4j.create(new double[y_size]);
            this.kernel = kernel;
            this.train_grid_n = train_grid_n;
            this.out_grid_n = out_grid_n;
            this.loss_fn = loss_fn;
            this.out = Nd4j.create(new double[y_size]);
            this.Name = "KDE NB";
        }

        private void calcH() {
            this.h_x = this.c_x.div(sqrt(this.n));
            this.h_y = this.c_y.div(sqrt(this.n));
        }

        private INDArray CVCE() {
            INDArray sum1 = zeros(this.x_size, this.y_size);
            for(int j = 0; j < this.n; j++) {
                INDArray sum2 = zeros(this.x_size, this.y_size);
                for(int i = 0; i < this.n; i++) {
                   if(i==j) {
                       continue;
                   }
                   sum2.addi(this.kernel.out(this.x_examples.getRow(j).sub(this.x_examples.getRow(i)).add(epsilon).div(this.h_x.add(epsilon))).add(epsilon).reshape(this.x_size,1).mmul(
                           this.kernel.out(this.y_examples.getRow(j).sub(this.y_examples.getRow(i)).add(epsilon).div(this.h_y.add(epsilon))).add(epsilon).reshape(1,this.y_size)));
                }
                sum1.addi(Nd4j.exec(new Log(Nd4j.exec(new Pow(this.h_x.mmul(this.h_y).mul(this.n - 1), -1)).mul(sum2))));
            }
            return sum1.mul((double) -1/this.n).sum(0).sum(0);
        }

        public void fit(INDArray input, INDArray output) {
            this.n = input.rows();
            this.x_examples = input;
            this.y_examples = output;

            JProgressBar pbar = new JProgressBar(0, 100);
            pbar.setStringPainted(true);
            this.panel.add(pbar);
            JTextPane timetext = new JTextPane();
            this.panel.add(timetext);
            this.frame.setTitle("KDE NB Fit Progress");
            this.frame.setVisible(true);

            long time_0 = java.time.Instant.now().toEpochMilli();

            // Grid Search
            boolean initialized = false;
            INDArray lowest_loss = ones(1);
            INDArray best_c_x = Nd4j.create(this.x_size);
            INDArray best_c_y = Nd4j.create(this.y_size);
            INDArray x_ind = zeros(x_size);
            boolean brkx = false;
            while(true) {
                INDArray y_ind = zeros(y_size);
                boolean brky = false;
                while(true) {
                    this.c_x = ones(this.x_size).div((double) this.train_grid_n).mul(x_ind.add(1));
                    this.c_y = ones(this.y_size).div((double) this.train_grid_n).mul(y_ind.add(1));
                    this.calcH();
                    INDArray this_loss = this.loss_fn != null ? this.loss_fn.evaluate(this.predict(input), output) : this.CVCE();
                    if(!initialized || this_loss.lt(lowest_loss).any()) {
                        initialized = true;
                        lowest_loss = this_loss;
                        best_c_x = this.c_x;
                        best_c_y = this.c_y;
                    }

                    if(brky) {
                        break;
                    }
                    for(int i = 0; i < this.y_size; i++) {
                        if(y_ind.getInt(i) < (this.train_grid_n-1)) {
                            y_ind.putScalar(i, y_ind.getInt(i)+1);
                            break;
                        }
                    }
                    if(y_ind.getInt(this.y_size-1) == (this.train_grid_n-1)) {
                        brky = true;
                    }
                }
                if(brkx) {
                    break;
                }
                for(int i = 0; i < this.x_size; i++) {
                    if(x_ind.getInt(i) < (this.train_grid_n-1)) {
                        x_ind.putScalar(i, x_ind.getInt(i)+1);
                        break;
                    }
                }
                if(x_ind.getInt(this.x_size-1) == (this.train_grid_n-1)) {
                    brkx = true;
                }
                double progress = (double) x_ind.sumNumber() / ((this.train_grid_n-1) * x_size);
                pbar.setValue((int) (100 * progress));
                timetext.setText(Long.toString((long) ((java.time.Instant.now().toEpochMilli() - time_0) * (1/progress - 1) / 1000)) + " seconds remaining");
            }
            this.c_x = best_c_x;
            this.c_y = best_c_y;
            calcH();
            pbar.setValue(100);
            this.frame.setVisible(false);
            this.panel.remove(pbar);
            this.panel.remove(timetext);
        }

        public void once(INDArray input) {
            INDArray best_y = zeros(this.y_size);
            INDArray best_prob = zeros(this.y_size);
            for(int j = 0; j < this.out_grid_n; j++) {
                INDArray y_test = ones(this.y_size).div((double) this.out_grid_n-1).mul(j);
                INDArray num = zeros(this.x_size, this.y_size);
                for (int i = 0; i < this.n; i++) {
                    num.addi(this.kernel.out(input.sub(this.x_examples.getRow(i)).add(epsilon).div(this.h_x.add(epsilon))).add(epsilon).mmul(
                            this.kernel.out(y_test.sub(this.y_examples.getRow(i)).add(epsilon).div(this.h_y.add(epsilon))).add(epsilon)));
                }
                INDArray den = zeros(this.y_size);
                for (int i = 0; i < this.n; i++) {
                    den.addi(this.kernel.out(y_test.sub(this.y_examples.getRow(i)).add(epsilon).div(this.h_y.add(epsilon))).add(epsilon));
                }
                INDArray result = num.prod(0).div(den).div(this.h_y);
                INDArray greater = result.gt(best_prob);
                for(int k = 0; k < this.y_size; k++) {
                    if(greater.getInt(k) == 1) {
                        best_y.put(k, y_test.getScalar(k));
                        best_prob.put(k, result.getScalar(k));
                    }
                }
            }
            this.out = best_y;
        }
    }
}