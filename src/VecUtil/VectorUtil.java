package VecUtil;

import java.util.ArrayList;

public class VectorUtil {
	public static ArrayList<Double> Add(ArrayList<Double> v1, ArrayList<Double> v2) {
		assert(v1.size() == v2.size());
		ArrayList<Double> v = new ArrayList<>();
		double norm = 0.0;
		for (int i = 0; i < v1.size(); i++) {
			double tmp = v1.get(i) + v2.get(i);
			norm += Math.pow(tmp, 2);
			v.add(tmp);
		}
		for (int i = 0; i < v.size(); i++) {
			v.set(i, v.get(i) / norm);
		}
		return v;
	}

	public static Double cosineSimilarity(ArrayList<Double> v1, ArrayList<Double> v2) {
		double dotProduct = 0.0;
		double norm1 = 0.0;
		double norm2 = 0.0;
		for (int i = 0; i < v1.size(); i++) {
			dotProduct += v1.get(i) * v2.get(i);
			norm1 += Math.pow(v1.get(i), 2);
			norm2 += Math.pow(v2.get(i), 2);
		}
		return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
	}
	
	public static double sigmoid(double x) {
	    return (1/( 1 + Math.pow(Math.E,(-1*x))));
	  }
}
