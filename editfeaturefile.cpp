#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>


#define outNum 500;

/**********************************************
This code will merge train.ark and train.label.
	output data format:
	Id0       
	frameNum0
	labels0
	features0
	Id1       
	frameNum1
	labels1
	features1	
***********************************************/

using namespace std;

string ALL[48]={ 	"aa", "ae", "ah", "ao", "aw", "ax",
			       	"ay",  "b", "ch", "cl",  "d", "dh",
			       	"dx", "eh", "el", "en","epi", "er",
	    	   		"ey",  "f",  "g", "hh", "ih", "ix",
	       	   		"iy", "jh",  "k",  "l",  "m", "ng",
	       	    	 "n", "ow", "oy",  "p",  "r", "sh",
	      	  	   "sil",  "s", "th",  "t", "uh", "uw",
	          	   "vcl",  "v",  "w",  "y", "zh",  "z" };

string uttereenceID(string str){
	bool two;
	size_t i = str.rfind("_");
	string ID;

	ID = str.substr(0,i);
	return ID;
}

int phone2int(string phone){
	for ( int j = 0; j < 48; j++ )
	{
		if ( phone == ALL[j] )
			return j;
	}	
}

int main()
{
	map<string, int> out;
	ifstream train("test.ark");
	//ifstream lab("../label/train.lab");

	string temp,features;
	string space = " ";
	size_t pos;
	int frameNum,totNum;
	vector<string> frameName;
	vector<string> content;
	vector<int> labels;
	vector<int> phones;
	
	//make the map from frameID to label.
	while(getline(lab,temp)){
		pos = temp.find(',');
		out[temp.substr(0,pos)] = phone2int( temp.substr(pos+1) );
	}
    
    //merge train.ark and train.lab
    int smallTest;
    smallTest = 0;
    totNum = 0;
	while(getline(train,temp)){
		pos = temp.find(' ');
		frameName.push_back( uttereenceID(temp.substr(0,pos)) );
		content.push_back( temp.substr(pos) );
		labels.push_back( out[temp.substr(0,pos)] );
		totNum++;
		smallTest++;
	}
    features = "";
    int c = 0;
	for(int i = 0; i < totNum-1; i++){
		if(frameName[i].compare(frameName[i+1])==0){
			features += space;
			features += content[i];
			phones.push_back(labels[i]);
			c++;
		}
		//final case
		else if(i == totNum-2){
			cout << frameName[i] << endl;
			features += space;
			features += content[i];
			phones.push_back(labels[i]);
			c++;
			features += space;
			features += content[i+1];
			phones.push_back(labels[i+1]);
			c++;
			cout << c <<endl;
			for(int j = 0; j < phones.size(); j++) cout << phones[j] << " ";
			cout << labels[totNum-1];
			cout << endl;
			cout << features <<endl;
			//cout << content[totNum-1];
			features = "";
			phones.clear();
			c = 0;
			break;
		}
		else{
			features += space;
			features += content[i];
			phones.push_back(labels[i]);
			c++;
			cout << frameName[i-1] << endl;
			cout << c << endl;
			for(int j = 0; j < phones.size(); j++) cout << phones[j] << " ";
			cout << endl;
			cout << features <<endl;
			features = "";
			phones.clear();
			c = 0;
		}


	}


	return 0;
}
