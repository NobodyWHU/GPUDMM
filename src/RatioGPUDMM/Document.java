package RatioGPUDMM;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class Document {
  
  public String [] words;
  public int id;
  public String category;
  
  
  public Document(int docid, String category, String [] words){
    this.id = docid;
    this.category = category;
    this.words = words;
  }
  
  public static ArrayList<Document> LoadCorpus(String filename){
	    try{
	      FileInputStream fis = new FileInputStream(filename);
	      InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
	      BufferedReader reader = new BufferedReader(isr);
	      String line;
	      ArrayList<Document> doc_list = new ArrayList();
	      while((line = reader.readLine()) != null){
	        line = line.trim();
	        String[] items = line.split("\t");
	        int docid = Integer.parseInt(items[0]);
	        String[] others = items[1].split("\\|");
	        String category = others[0];
	        String words_str = others[1].trim();
	        String[] words = words_str.split("\\s");
	        Document doc = new Document(docid, category, words);
	        doc_list.add(doc);
	      }
	      return doc_list;
	    }
	    catch (Exception e){
	      System.out.println("Error while reading other file:" + e.getMessage());
	      e.printStackTrace();
//	      return false;
	  }
	    return null;
	    
	  }

  public static void main(String[] args) {
    // TODO 自动生成的方法存根
    String [] sarray = {"科技","专科","西安","大学"};
    Document doc = new Document(1, "院校信息", sarray);
    System.out.println(doc.id);
  }
}
