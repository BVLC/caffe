#include "caffe/util/tree.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

tree::tree()
{

}

tree::tree(string filename)
{
  //char file[20] = filename.c_str();
  //std::cout<< "filename: " << filename <<std::endl;
  ifstream fin(filename.c_str());
  if(!fin)
  {
      cout<<"file is not exist!"<<endl;
  }

  string line;
  int last_parent = -1;
  int group_size = 0;
  int groups = 0;
  int n = 0;
  
  while(getline(fin, line)){
    string id = line.substr(0, 9);
    int parent = -1;
    
    stringstream ss;
    ss << line.substr(10).c_str();
    ss >> parent;
    //parent = std::atoi(line.substr(10).c_str());
    //std::cout<< line << " p:" << parent <<std::endl;

    this->parent.push_back(parent);
    this->child.push_back(-1);
    this->name.push_back(id);

    if (parent != last_parent)
    {
        ++groups;
        this->group_offset.push_back(n - group_size);
        this->group_size.push_back(group_size);
        group_size = 0;
        last_parent = parent;
    }
    this->group.push_back(groups);
    if (parent >= 0)
    {
        this->child[parent] = groups;
    }
    ++n;
    ++group_size;
  }
  ++groups;
  this->group_offset.push_back(n-group_size);
  this->group_size.push_back(group_size);
  this->n = n;
  this->groups = groups;

  for (int i = 0; i < n; ++ i)
    this->leaf.push_back(1);
  for (int i = 0; i < n; ++ i)
      if (this->parent[i] >= 0)
          this->leaf[this->parent[i]] = 0;

  fin.close();
  //std::cout<< "groups: "<< groups << "\t" << this->groups << std::endl;
}
