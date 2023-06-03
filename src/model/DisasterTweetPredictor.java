package model;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial ;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class DisasterTweetPredictor {

	//train ARFF path
	private static String ARFF_FILE = "train.arff";
	private static String DICTIONARY = "dictionarylist";

	private Classifier model;
	private StringToWordVector filter;

	public DisasterTweetPredictor() {

		//set filter
		try {
			this.filter = new StringToWordVector();

			filter.setOptions(Utils.splitOptions("-R first -P wv- -W 10000 -prune-rate -1.0 -T -N 0 -L -stemmer weka.core.stemmers.LovinsStemmer -stopwords-handler weka.core.stopwords.Rainbow -M 1 -tokenizer weka.core.tokenizers.TweetNLPTokenizer -dictionary \"" + DICTIONARY + "\""));
			filter.setDoNotCheckCapabilities(true);
		} catch(Exception e) {
			e.printStackTrace();
		}
		
		//train();

		//set model
		try {
			this.model = (Classifier) weka.core.SerializationHelper.read("default.model");
		} catch (Exception e) {
			train(); //if no model generated yet
		}
	}


	//Read ARFF
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		//To read data file
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
			ex.printStackTrace();
		}
		return inputReader;
	}

	//formats data to word vectors
	public Instances stringToWordVector(Instances raw, boolean isPredict) throws Exception {
		if(isPredict == false) {
			filter.setInputFormat(raw);
			raw = Filter.useFilter(raw, filter);
		} else {
			FixedDictionaryStringToWordVector dictVector = new FixedDictionaryStringToWordVector();
			dictVector.setOptions(Utils.splitOptions("-T -R first -dictionary " + DICTIONARY + " -L -stemmer weka.core.stemmers.NullStemmer -stopwords-handler \"weka.core.stopwords.Rainbow \" -tokenizer \"weka.core.tokenizers.TweetNLPTokenizer \""));
			dictVector.setInputFormat(raw);
			raw = Filter.useFilter(raw, dictVector);
		}
		Reorder reorder = new Reorder();
		reorder.setOptions(Utils.splitOptions("-R 2-last,first"));
		reorder.setInputFormat(raw);
		raw = Filter.useFilter(raw, reorder);
		//for debugging
		//System.out.println(raw);
		return raw;
	}

	//Train new model (or retrain model)
	public void train() {
		Instances data;
		BufferedReader datafile = readDataFile(ARFF_FILE); //make sure to put updated ARFF file (if any)
		try {
			ArffReader arff = new ArffReader(datafile);
			Instances raw = arff.getData();

			data = stringToWordVector(raw, false);
			data.setClassIndex(data.numAttributes() - 1);

			NaiveBayesMultinomial model = new NaiveBayesMultinomial();
			//Options for classifier would go here.
			//model.setOptions(Utils.splitOptions("quotedOptionString"));
			data.setClassIndex(data.numAttributes() - 1);
			model.buildClassifier(data);
			weka.core.SerializationHelper.write("default.model", model);
			this.model = model;

		} catch(Exception e) {
			System.out.println("No train.arff found.");
			e.printStackTrace();
		}
	}


	//Predicts data using saved model.
	//Output result as string.
	//0 means not related to disaster, 1 means related to disaster.
	public String predict(String dataRaw) {
		System.out.println("Text: " + dataRaw);

		//Build query
		ArrayList<Attribute> atts = new ArrayList<Attribute>(2);
		Attribute textAtt = new Attribute("text", true);
		ArrayList<String> classVal1 = new ArrayList<String>();
		classVal1.add("0");
		classVal1.add("1");
		Attribute targetAtt = new Attribute("target", classVal1);

		atts.add(textAtt);
		atts.add(targetAtt);

		Instances data = new Instances("tweet",atts,0);

		Instance inst = new DenseInstance(2);
		inst.setValue(textAtt, dataRaw);
		inst.setValue(targetAtt, 1);
		data.add(inst);
		System.out.println(data);
		
		Instances newData;
		try {
			newData = stringToWordVector(data, true);
		} catch (Exception e1) {
			System.out.println(e1);
			return "Could not process tweet.";
		}
		

		//predict query
		
		System.out.println(newData.instance(0));
		newData.setClassIndex(newData.numAttributes() - 1);
		Instances resultInstance = new Instances(newData);
		Instance prediction = null;
		String predString = null;
		
		
		try {
			Double result = model.classifyInstance(newData.firstInstance());
			
			resultInstance.instance(0).setClassValue(result);
			System.out.println(resultInstance.instance(0));
			prediction = resultInstance.instance(0);
			predString = prediction.stringValue(newData.numAttributes() - 1);
			System.out.println("prediction: " + predString);
			return Double.toString(result);

		} catch(Exception e) {
			e.printStackTrace();
			return "ERROR";
		}	
	}

	//test this predictor
	public static void main(String[] args) {
		DisasterTweetPredictor predictor = new DisasterTweetPredictor();

		Scanner sc = new Scanner(System.in);
		System.out.println("Insert a tweet: ");
		String tweet = sc.nextLine();
		sc.close();

		System.out.println("Output to user: " + predictor.predict(tweet));
	}
}
