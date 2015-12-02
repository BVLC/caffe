import java.io.*;
import java.util.*;

public class ReadProperties {
	public static void main(String[] args) throws IOException {
		if(args.length <= 0) { System.out.println("No file provided."); return; }

		File f = new File(args[0]);

		if(!f.exists()) { System.out.println("File not found: " + args[0]); return; }

		Properties prop = new Properties();
		prop.load(new FileInputStream(f));

		boolean isFirst = true; // I fucking hate java, why don't they have a native string join function?
		System.out.print("{");
		for (Map.Entry<Object, Object> item : prop.entrySet()) {
			String key = (String) item.getKey();
			String value = (String) item.getValue();
			
			if(isFirst) { isFirst = false; }
			else { System.out.print(","); }

			System.out.print("\"" + escape(key) + "\":\"" + escape(value) + "\"");
		}
		System.out.print("}");
	}

	static String escape(String s) { // Taken from http://code.google.com/p/json-simple/
		StringBuffer sb = new StringBuffer();
		for(int i = 0; i < s.length(); i++) {
			char ch = s.charAt(i);
			switch(ch) {
				case '"': sb.append("\\\""); break;
				case '\\': sb.append("\\\\"); break;
				case '\b': sb.append("\\b"); break;
				case '\f': sb.append("\\f"); break;
				case '\n': sb.append("\\n"); break;
				case '\r': sb.append("\\r"); break;
				case '\t': sb.append("\\t"); break;
				case '/': sb.append("\\/"); break;
				default:
				//Reference: http://www.unicode.org/versions/Unicode5.1.0/
				if (('\u0000' <= ch && ch <= '\u001F') 
					|| ('\u007F' <= ch && ch <= '\u009F') 
					|| ('\u2000' <= ch && ch <= '\u20FF')) {
					String ss = Integer.toHexString(ch);
					sb.append("\\u");
					for(int k = ss.length(); k < 4; k++) {
						sb.append('0');
					}
					sb.append(ss.toUpperCase());
				} else {
					sb.append(ch);
				}
			}
		}

		return sb.toString();
	}
}