package RatioGPUDMM;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import RatioGPUDMM.Document;
import RatioGPUDMM.TopicalWordComparator;

public class RatioGPUDMM {
	public Set<String> wordSet;
	public int numTopic;
	public double alpha, beta;
	public int numIter;
	public int saveStep;
	public ArrayList<Document> docList;
	public int roundIndex;
	private Random rg;
	public double threshold;
	public double weight;
	public int topWords;
	public int filterSize;
	public String word2idFileName;
	public String similarityFileName;

	public Map<String, Integer> word2id;
	public Map<Integer, String> id2word;
	public Map<Integer, Double> wordIDFMap;
	public Map<Integer, Map<Integer, Double>> docUsefulWords;
	public ArrayList<ArrayList<Integer>> topWordIDList;
	public int vocSize;
	public int numDoc;
	private double[][] schema;
	public ArrayList<int[]> docToWordIDList;
	public String initialFileName;  // we use the same initial for DMM-based model
	public double[][] phi;
	private double[] pz;
	private double[][] pdz;
	private double[][] topicProbabilityGivenWord;

	public ArrayList<ArrayList<Boolean>> wordGPUFlag; // wordGPUFlag.get(doc).get(wordIndex)
	public int[] assignmentList; // topic assignment for every document
	public ArrayList<ArrayList<Map<Integer, Double>>> wordGPUInfo;

	private int[] mz; // have no associatiom with word and similar word
	private double[] nz; // [topic]; nums of words in every topic
	private double[][] nzw; // V_{.k}
	private Map<Integer, Map<Integer, Double>> schemaMap;

	public RatioGPUDMM(ArrayList<Document> doc_list, int num_topic, int num_iter, int save_step, double beta,
			double alpha, double threshold) {
		docList = doc_list;
		numDoc = docList.size();
		numTopic = num_topic;
		this.alpha = alpha;
		numIter = num_iter;
		saveStep = save_step;
		this.threshold = threshold;
		this.beta = beta;

	}

	public boolean loadWordMap(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			
			//construct word2id map
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(",");
				word2id.put(items[0], Integer.parseInt(items[1]));
				id2word.put(Integer.parseInt(items[1]), items[0]);
			}
			System.out.println("finish read wordmap and the num of word is " + word2id.size());
			return true;
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
	}

	/**
	 * Collect the similar words Map, not including the word itself
	 * 
	 * @param filename:
	 *            shcema_similarity filename
	 * @param threshold:
	 *            if the similarity is bigger than threshold, we consider it as
	 *            similar words
	 * @return
	 */
	public Map<Integer, Map<Integer, Double>> loadSchema(String filename, double threshold) {
		int word_size = word2id.size();
		Map<Integer, Map<Integer, Double>> schemaMap = new HashMap<Integer, Map<Integer, Double>>();
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			int lineIndex = 0;
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");

				for (int i = 0; i < items.length; i++) {
					Double value = Double.parseDouble(items[i]);
					schema[lineIndex][i] = value;
				}
				lineIndex++;
				if (lineIndex % 500 == 0) {
					System.out.println(lineIndex);
				}
			}
			double count = 0.0;
			for (int i = 0; i < word_size; i++) {
				Map<Integer, Double> tmpMap = new HashMap<Integer, Double>();
				for (int j = 0; j < word_size; j++) {
					double v = schema[j][i];
					if (Double.compare(v, threshold) > 0) {
						tmpMap.put(j, v);
					}
				}
				if (tmpMap.size() > filterSize) {
					tmpMap.clear();
				}
				tmpMap.remove(i);
				if (tmpMap.size() == 0) {
					continue;
				}
				count += tmpMap.size();
				schemaMap.put(i, tmpMap);
			}
			System.out.println("finish read schema, the avrage number of value is " + count / schemaMap.size());
			return schemaMap;
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * 
	 * @param wordID
	 * @param topic
	 * @return word probability given topic 
	 */
	public double getWordProbabilityUnderTopic(int wordID, int topic) {
		return (nzw[topic][wordID] + beta) / (nz[topic] + vocSize * beta);
	}

	public double getMaxTopicProbabilityForWord(int wordID) {
		double max = -1.0;
		for (int t = 0; t < numTopic; t++) {
			double tmp = getWordProbabilityUnderTopic(wordID, t);
			if (Double.compare(tmp, max) > 0) {
				max = tmp;
			}
		}
		return max;
	}

	/**
	 * Get the top words under each topic given current Markov status.
	 * not used in this RatioGPUDMM
	 */
	private ArrayList<ArrayList<Integer>> getTopWordsUnderEachTopicGivenCurrentMarkovStatus() {
		compute_pz();
		compute_phi();
		if (topWordIDList.size() <= numTopic) {
			for (int t = 0; t < numTopic; t++) {
				topWordIDList.add(new ArrayList<>());
			}
		}
		int top_words = topWords;

		for (int t = 0; t < numTopic; ++t) {
			Comparator<Integer> comparator = new TopicalWordComparator(phi[t]);
			PriorityQueue<Integer> pqueue = new PriorityQueue<Integer>(top_words, comparator);

			for (int w = 0; w < vocSize; ++w) {
				if (pqueue.size() < top_words) {
					pqueue.add(w);
				} else {
					if (phi[t][w] > phi[t][pqueue.peek()]) {
						pqueue.poll();
						pqueue.add(w);
					}
				}
			}

			ArrayList<Integer> oneTopicTopWords = new ArrayList<>();
			while (!pqueue.isEmpty()) {
				oneTopicTopWords.add(pqueue.poll());
			}
			topWordIDList.set(t, oneTopicTopWords);
		}
		return topWordIDList;
	}

	/**
	 * update the p(z|w) for every iteration
	 */
	public void updateTopicProbabilityGivenWord() {
		// TODO we should update pz and phi information before
		compute_pz();
		compute_phi();  //update p(w|z)
		for (int i = 0; i < vocSize; i++) {
			double row_sum = 0.0;
			for (int j = 0; j < numTopic; j++) {
				topicProbabilityGivenWord[i][j] = pz[j] * phi[j][i];
				row_sum += topicProbabilityGivenWord[i][j];
			}
			for (int j = 0; j < numTopic; j++) {
				topicProbabilityGivenWord[i][j] = topicProbabilityGivenWord[i][j] / row_sum;  //This is p(z|w)
			}
		}
	}
	
	
	public double findTopicMaxProbabilityGivenWord(int wordID) {
		double max = -1.0;
		for (int i = 0; i < numTopic; i++) {
			double tmp = topicProbabilityGivenWord[wordID][i];
			if (Double.compare(tmp, max) > 0) {
				max = tmp;
			}
		}
		return max;
	}

	public double getTopicProbabilityGivenWord(int topic, int wordID) {
		return topicProbabilityGivenWord[wordID][topic];
	}
	
	/**
	 * update GPU flag, which decide whether do GPU operation or not
	 * @param docID
	 * @param newTopic
	 */
	public void updateWordGPUFlag(int docID, int newTopic) {
		// we calculate the p(t|w) and p_max(t|w) and use the ratio to decide we
		// use gpu for the word under this topic or not
		int[] termIDArray = docToWordIDList.get(docID);
		ArrayList<Boolean> docWordGPUFlag = new ArrayList<>();
		for (int t = 0; t < termIDArray.length; t++) {
			
			int termID = termIDArray[t];
			double maxProbability = findTopicMaxProbabilityGivenWord(termID);
			double ratio = getTopicProbabilityGivenWord(newTopic, termID) / maxProbability;

			double a = rg.nextDouble();
			docWordGPUFlag.add(Double.compare(ratio, a) > 0);
		}
		wordGPUFlag.set(docID, docWordGPUFlag);
	}

	/**
	 * 
	 * @param filename for topic assignment for each document
	 */
	public void loadInitialStatus(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");
				assert(items.length == assignmentList.length);
				for (int i = 0; i < items.length; i++) {
					assignmentList[i] = Integer.parseInt(items[i]);
				}
				break;
			}

			System.out.println("finish loading initial status");
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
		}
	}

	/**
	 * change the counter of this model
	 * @param topic
	 * @param wordID
	 * @param docID
	 * @param flag  add or subtract
	 * @param gpuFlag
	 */
	public void doCounterForWord(int topic, int wordID, Integer docID, int flag, Boolean gpuFlag) {
		// we alwayls change the original word
		nzw[topic][wordID] += flag;
		nz[topic] += flag;

		if (gpuFlag) {
			if (schemaMap.containsKey(wordID)) {
				Map<Integer, Double> valueMap = schemaMap.get(wordID);
				// update the counter
				for (Map.Entry<Integer, Double> entry : valueMap.entrySet()) {
					Integer k = entry.getKey();
					double pt_w = getTopicProbabilityGivenWord(topic, k);
					double pMax_w = findTopicMaxProbabilityGivenWord(k);
					double ratio = pt_w / pMax_w;
					double a = rg.nextDouble();
					if (Double.compare(ratio, a) > 0) { // do gpu operation
						double v = weight;
						nzw[topic][k] += flag * v;
						nz[topic] += flag * v;
					} else {
						// we do nothing!
					}

				}
				// System.out.println("############");
			}
		}

	}
	
	
	public void updateCount(Integer topic, Integer docID, int[] termIDArray, Integer flag) {
		mz[topic] += flag;

		// we update gpu flag for every document before it change the counter
		// when adding numbers
		if (flag > 0) {
			updateWordGPUFlag(docID, topic);
		}
		for (int t = 0; t < termIDArray.length; t++) {
			int wordID = termIDArray[t];
			boolean gpuFlag = wordGPUFlag.get(docID).get(t);
			doCounterForWord(topic, wordID, docID, flag, gpuFlag);
		}
	}

	public void ratioCount(Integer topic, Integer docID, int[] termIDArray, int flag) {
		mz[topic] += flag;
		for (int t = 0; t < termIDArray.length; t++) {
			int wordID = termIDArray[t];
			nzw[topic][wordID] += flag;
			nz[topic] += flag;
		}
		// we update gpu flag for every document before it change the counter
		// when adding numbers
		if (flag > 0) {
			updateWordGPUFlag(docID, topic);
			for (int t = 0; t < termIDArray.length; t++) {
				int wordID = termIDArray[t];
				boolean gpuFlag = wordGPUFlag.get(docID).get(t);
				Map<Integer, Double> gpuInfo = new HashMap<>();
				if (gpuFlag) { // do gpu count
					if (schemaMap.containsKey(wordID)) {
						Map<Integer, Double> valueMap = schemaMap.get(wordID);
						// update the counter
						for (Map.Entry<Integer, Double> entry : valueMap.entrySet()) {
							Integer k = entry.getKey();
							double v = weight;
							nzw[topic][k] += v;
							nz[topic] += v;
							gpuInfo.put(k, v);
						} // end loop for similar words
					} else { // schemaMap don't contain the word

						// the word doesn't have similar words, the infoMap is empty
						// we do nothing
					}
				} else { // the gpuFlag is False
					// it means we don't do gpu, so the gouInfo map is empty
				}
				wordGPUInfo.get(docID).set(t, gpuInfo); // we update the gpuinfo
														// map
			}
		} else { // we do subtraction according to last iteration information
			for (int t = 0; t < termIDArray.length; t++) {
				Map<Integer, Double> gpuInfo = wordGPUInfo.get(docID).get(t);
				int wordID = termIDArray[t];
				// boolean gpuFlag = wordGPUFlag.get(docID).get(t);
				if (gpuInfo.size() > 0) {
					for (int similarWordID : gpuInfo.keySet()) {
						// double v = gpuInfo.get(similarWordID);
						double v = weight;
						nzw[topic][similarWordID] -= v;
						nz[topic] -= v;
						// if(Double.compare(0, nzw[topic][wordID]) > 0){
						// System.out.println( nzw[topic][wordID]);
						// }
					}
				}
			}
		}

	}

	public void normalCount(Integer topic, int[] termIDArray, Integer flag) {
		mz[topic] += flag;
		for (int t = 0; t < termIDArray.length; t++) {
			int wordID = termIDArray[t];
			nzw[topic][wordID] += flag;
			nz[topic] += flag;
		}
	}
	
	
	public void initNewModel() {
		wordGPUFlag = new ArrayList<>();
		docToWordIDList = new ArrayList<int[]>();
		word2id = new HashMap<String, Integer>();
		id2word = new HashMap<Integer, String>();
		wordIDFMap = new HashMap<Integer, Double>();
		docUsefulWords = new HashMap<Integer, Map<Integer, Double>>();
		wordSet = new HashSet<String>();
		topWordIDList = new ArrayList<>();
		assignmentList = new int[numDoc];
		wordGPUInfo = new ArrayList<>();
		rg = new Random();
		// construct vocabulary
		loadWordMap(word2idFileName);

		vocSize = word2id.size();
		phi = new double[numTopic][vocSize];
		pz = new double[numTopic];
		pdz = new double[numDoc][numTopic];

		schema = new double[vocSize][vocSize];
		topicProbabilityGivenWord = new double[vocSize][numTopic];

		for (int i = 0; i < docList.size(); i++) {
			Document doc = docList.get(i);
			int[] termIDArray = new int[doc.words.length];
			ArrayList<Boolean> docWordGPUFlag = new ArrayList<>();
			ArrayList<Map<Integer, Double>> docWordGPUInfo = new ArrayList<>();
			for (int j = 0; j < doc.words.length; j++) {
				termIDArray[j] = word2id.get(doc.words[j]);
				docWordGPUFlag.add(false); // initial for False for every word
				docWordGPUInfo.add(new HashMap<Integer, Double>());
			}
			wordGPUFlag.add(docWordGPUFlag);
			wordGPUInfo.add(docWordGPUInfo);
			docToWordIDList.add(termIDArray);
		}

		// init the counter
		mz = new int[numTopic];
		nz = new double[numTopic];
		nzw = new double[numTopic][vocSize];
	}

	public void init_GSDMM() {
//		 schemaMap = loadSchema("E:\\pythonWorkspace\\GPUBTM\\data\\qa_word_similarity.txt",threshold);
		schemaMap = loadSchema(similarityFileName, threshold);
		loadInitialStatus(initialFileName);

		for (int d = 0; d < docToWordIDList.size(); d++) {
			int[] termIDArray = docToWordIDList.get(d);
			int topic = assignmentList[d];
//			 int topic = rg.nextInt(numTopic);
//			 assignmentList[d] = topic;
			normalCount(topic, termIDArray, +1);
		}
		System.out.println("finish init_MU!");
	}

	private static long getCurrTime() {
		return System.currentTimeMillis();
	}

	public void run_iteration() {

		for (int iteration = 1; iteration <= numIter; iteration++) {
			System.out.println(iteration + "th iteration begin");

			long _s = getCurrTime();
			// getTopWordsUnderEachTopicGivenCurrentMarkovStatus();
			updateTopicProbabilityGivenWord();
			for (int s = 0; s < docToWordIDList.size(); s++) {

				int[] termIDArray = docToWordIDList.get(s);
				int preTopic = assignmentList[s];

				ratioCount(preTopic, s, termIDArray, -1);

				double[] pzDist = new double[numTopic];
				for (int topic = 0; topic < numTopic; topic++) {
					double pz = 1.0 * (mz[topic] + alpha) / (numDoc - 1 + numTopic * alpha);
					double value = 1.0;
					double logSum = 0.0;
					for (int t = 0; t < termIDArray.length; t++) {
						int termID = termIDArray[t];
						value *= (nzw[topic][termID] + beta) / (nz[topic] + vocSize * beta + t);
						// we do not use log, it is a little slow
						// logSum += Math.log(1.0 * (nzw[topic][termID] + beta) / (nz[topic] + vocSize * beta + t));
					}
//					value = pz * Math.exp(logSum);
					value = pz * value;
					pzDist[topic] = value;
				}

				for (int i = 1; i < numTopic; i++) {
					pzDist[i] += pzDist[i - 1];
				}

				double u = rg.nextDouble() * pzDist[numTopic - 1];
				int newTopic = -1;
				for (int i = 0; i < numTopic; i++) {
					if (Double.compare(pzDist[i], u) >= 0) {
						newTopic = i;
						break;
					}
				}
				// update
				assignmentList[s] = newTopic;
				ratioCount(newTopic, s, termIDArray, +1);

			}
			long _e = getCurrTime();
			System.out.println(iteration + "th iter finished and every iterration costs " + (_e - _s) + "ms! Snippet "
					+ numTopic + " topics round " + roundIndex);
		}
	}

	public void run_GSDMM(String flag) {
		initNewModel();
		init_GSDMM();
		run_iteration();
		saveModel(flag);
	}

	public void saveModel(String flag) {

		compute_phi();
		compute_pz();
		compute_pzd();
		saveModelPz(flag + "_theta.txt");
		saveModelPhi(flag + "_phi.txt");
		saveModelWords(flag + "_words.txt");
		saveModelAssign(flag + "_assign.txt");
		saveModelPdz(flag + "_pdz.txt");
	}

	public void compute_phi() {
		for (int i = 0; i < numTopic; i++) {
			double sum = 0.0;
			for (int j = 0; j < vocSize; j++) {
				sum += nzw[i][j];
			}
			for (int j = 0; j < vocSize; j++) {
				phi[i][j] = (nzw[i][j] + beta) / (sum + vocSize * beta);
			}
		}
	}

	public void compute_pz() {
		double sum = 0.0;
		for (int i = 0; i < numTopic; i++) {
			sum += nz[i];
		}
		for (int i = 0; i < numTopic; i++) {
			pz[i] = 1.0 * (nz[i] + alpha) / (sum + numTopic * alpha);
		}
	}

	public void compute_pzd() {
		double[][] pwz = new double[vocSize][numTopic]; // pwz[word][topic]
		for (int i = 0; i < vocSize; i++) {
			double row_sum = 0.0;
			for (int j = 0; j < numTopic; j++) {
				pwz[i][j] = pz[j] * phi[j][i];
				row_sum += pwz[i][j];
			}
			for (int j = 0; j < numTopic; j++) {
				pwz[i][j] = pwz[i][j] / row_sum;
			}

		}

		for (int i = 0; i < numDoc; i++) {
			int[] doc_word_id = docToWordIDList.get(i);
			double row_sum = 0.0;
			for (int j = 0; j < numTopic; j++) {

				for (int wordID : doc_word_id) {
					pdz[i][j] += pwz[wordID][j];
				}
				row_sum += pdz[i][j];

			}
			for (int j = 0; j < numTopic; j++) {
				pdz[i][j] = pdz[i][j] / row_sum;
			}
		}
	}

	public boolean saveModelAssign(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numDoc; i++) {
				int topic = assignmentList[i];
				for (int j = 0; j < numTopic; j++) {
					int value = -1;
					if (j == topic) {
						value = 1;
					} else {
						value = 0;
					}
					out.print(value + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving assign list: " + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPdz(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numDoc; i++) {
				for (int j = 0; j < numTopic; j++) {
					out.print(pdz[i][j] + " ");
				}
				out.println();
			}

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving p(z|d) distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPz(String filename) {
		// return false;
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numTopic; i++) {
				out.print(pz[i] + " ");
			}
			out.println();

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving pz distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPhi(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numTopic; i++) {
				for (int j = 0; j < vocSize; j++) {
					out.print(phi[i][j] + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelWords(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename, "UTF8");
			for (String word : word2id.keySet()) {
				int id = word2id.get(word);
				out.println(word + "," + id);
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saveing words list: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public static void main(String[] args) {

		ArrayList<Document> doc_list = Document.LoadCorpus("../data/qa_data.txt");
		//here
		int num_iter = 1000, save_step = 200;
		double beta = 0.1;
		String similarityFileName = "../data/snippet_word_similarity.txt";
		double weight = 0.1;
		double threshold = 0.7;
		int filterSize = 20;
		
		for (int round = 1; round <= 5; round += 1) {
			for (int num_topic = 80; num_topic <= 80; num_topic += 20) {
				String initialFileName = "../data/topic" + num_topic + "_snippet_200iter_initial_status.txt";
//				String initialFileName = "../data/topic" + num_topic + "_qa_random_initial_status.txt";
				double alpha = 1.0 * 50 / num_topic;
				RatioGPUDMM gsdmm = new RatioGPUDMM(doc_list, num_topic, num_iter, save_step, beta, alpha, threshold);
				gsdmm.word2idFileName = "../data/qa_word2id.txt";
				gsdmm.topWords = 100;
				
				//here
				gsdmm.filterSize = filterSize;
				gsdmm.roundIndex = round;
				gsdmm.initialFileName = initialFileName;
				gsdmm.similarityFileName = similarityFileName;
				gsdmm.weight = weight;
				gsdmm.initNewModel();
				gsdmm.init_GSDMM();
				gsdmm.run_iteration();
				String flag = round+"round_"+num_topic + "topic_weight05_snippet" + "_filter20_iter1000_gpudmm";
				flag = "snippetUpdateNow/" + flag;
				gsdmm.saveModel(flag);
			}
		}
	}

}

/**
 * Comparator to rank the words according to their probabilities.
 */
class TopicalWordComparator implements Comparator<Integer> {
	private double[] distribution = null;

	public TopicalWordComparator(double[] distribution2) {
		distribution = distribution2;
	}

	@Override
	public int compare(Integer w1, Integer w2) {
		if (distribution[w1] < distribution[w2]) {
			return -1;
		} else if (distribution[w1] > distribution[w2]) {
			return 1;
		}
		return 0;
	}
}
