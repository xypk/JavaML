package mlutils.java;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class DataHandler {
    private String _srcdir = "";
    private Random random = new Random();
    private List<List<Writable>> rr_list = null;
    private int numRows;
    public RecordReader Reader = null;

    public DataHandler() {}
    public DataHandler(String srcdir) {
        this._srcdir = srcdir;
    }

    private Path getFullPath(String relative_path) {
        return Paths.get(this._srcdir, relative_path);
    }

    private DataHandler duplicate() {
        DataHandler copy = new DataHandler(this._srcdir);
        copy.random = this.random;
        copy.rr_list = this.rr_list;
        copy.numRows = this.numRows;
        copy.Reader = this.Reader;
        return copy;
    }

    public void setRandom(Random random) {
        this.random = random;
    }

    public int getRows() {
        return this.numRows;
    }

    public void shuffle() {
        Collections.shuffle(this.rr_list, this.random);
        this.Reader = new CollectionRecordReader(this.rr_list);
    }

    public void loadFromCSV(String relative_path, TransformProcess tp, int skip_num_lines, boolean shuffle) {
        RecordReader rr = this.Reader == null ? new CSVRecordReader(skip_num_lines) : this.Reader;

        rr = new TransformProcessRecordReader(rr, tp);

        try {
            rr.initialize(new FileSplit(new File(getFullPath(relative_path).toString())));

        } catch (IOException e) {
            System.err.println("IO Exception: " + e.getMessage());
        } catch (InterruptedException e) {
            System.err.println("Interrupted Exception: " + e.getMessage());
        }

        this.rr_list = rr.next(1);
        while(rr.hasNext()) {
            this.rr_list.add(rr.next());
        }

        this.numRows = this.rr_list.size();
        this.Reader = rr;

        if(shuffle) {
            this.shuffle();
        } else {
            this.Reader.reset();
        }
    }

    public void setRange(int[] usableRange) {
        this.rr_list = null;
        this.Reader.reset();
        if(usableRange[0] > 0) {
            for (int i = 0; i < usableRange[0]; i++) {
                this.Reader.next();
            }
        }
        for (int i = usableRange[0]; i <= usableRange[1]; i++) {
            if (this.Reader.hasNext()) {
                if (this.rr_list == null) {
                    this.rr_list = new ArrayList<>();
                }
                this.rr_list.add(this.Reader.next());
            } else {
                break;
            }
        }
        this.numRows = this.rr_list.size();
        this.Reader = new CollectionRecordReader(this.rr_list);
    }

    public DataHandler[] split(double train_ratio) {
        int num_train = (int) (this.getRows() * train_ratio);
        //int num_test = this.getRows() - num_train;
        DataHandler train_set = this.duplicate();
        DataHandler test_set = this.duplicate();
        train_set.setRange(new int[] {0, num_train-1});
        test_set.setRange(new int[] {num_train, this.getRows()-1});
        return new DataHandler[] {train_set, test_set};
    }

    public static void saveArray(String output_path, INDArray array, String header) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(output_path))) {
            writer.write(String.join(",", header));
            writer.newLine();

            for (long i = 0; i < array.rows(); i++) {
                StringBuilder line = new StringBuilder();

                for (long j = 0; j < array.columns(); j++) {
                    line.append(array.getDouble(i, j));

                    if (j < array.columns() - 1) {
                        line.append(",");
                    }
                }
                writer.write(line.toString());
                writer.newLine();
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static JsonNode readJson(String file_path) {
        JsonNode json_data = null;
        try {
            String json_string = new String(Files.readAllBytes(Paths.get(file_path)));
            ObjectMapper mapper = new ObjectMapper();
            json_data = mapper.readTree(json_string);
        } catch(IOException e) {
                e.printStackTrace();
        }
        return json_data;
    }

    public static DataHandler fromJson(String file_path) {
        JsonNode json_data = DataHandler.readJson(file_path);
        DataHandler DH = new DataHandler(json_data.get("srcdir").asText());
        DH.setRandom(new Random(json_data.get("seed").asInt()));
        
        Schema.Builder DataSchemaBuilder = new Schema.Builder();
        Iterator columns = json_data.get("columns").elements();
        while(columns.hasNext()) {
            JsonNode columnData = (JsonNode) columns.next();
            Iterator columnNames = columnData.get("names").elements();
            while(columnNames.hasNext()) {
                String columnName = ((JsonNode) columnNames.next()).asText();
                switch (columnData.get("type").asText()) {
                    case "integer":
                        DataSchemaBuilder = DataSchemaBuilder.addColumnInteger(columnName);
                        break;
                    case "double":
                        DataSchemaBuilder = DataSchemaBuilder.addColumnDouble(columnName);
                        break;
                    case "float":
                        DataSchemaBuilder = DataSchemaBuilder.addColumnFloat(columnName);
                        break;
                    case "long":
                        DataSchemaBuilder = DataSchemaBuilder.addColumnLong(columnName);
                        break;
                    case "string":
                        DataSchemaBuilder = DataSchemaBuilder.addColumnString(columnName);
                        break;
                    case "boolean":
                        DataSchemaBuilder = DataSchemaBuilder.addColumnBoolean(columnName);
                        break;
                }
            }
        }
        Schema DataSchema = DataSchemaBuilder.build();
        TransformProcess DataTransformProcess = new TransformProcess.Builder(DataSchema).build();
        DH.loadFromCSV(json_data.get("path").asText(), DataTransformProcess, json_data.get("skip_lines").asInt(), json_data.get("shuffle").asBoolean());
        return DH;
    }

    public static void setAttributes(Object clazz, Iterator<Map.Entry<String, JsonNode>> params) throws IllegalAccessException, NoSuchFieldException {
        while (params.hasNext()) {
            Map.Entry<String, JsonNode> next_entry = params.next();
            Field declaredField = clazz.getClass().getDeclaredField(next_entry.getKey());
            declaredField.set(clazz, (new ObjectMapper()).convertValue(next_entry.getValue(), declaredField.getType()));
        }
    }
}