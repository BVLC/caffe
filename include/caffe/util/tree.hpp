#ifndef TREE_H
#define TREE_H
#include <vector>
#include <string>

using std::vector;
using std::string;

class tree
{
public:
  tree();
  tree(string filename); 
  
  vector<int> leaf;
  int n;
  vector<int> parent;
  vector<int> child;
  vector<int> group;
  vector<string> name;
  
  int groups;
  vector<int> group_size;
  vector<int> group_offset;
};


#endif // TREE_H
