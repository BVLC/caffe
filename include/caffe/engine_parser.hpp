#ifndef CAFFE_MKLDNN_ENGINES_HPP_
#define CAFFE_MKLDNN_ENGINES_HPP_

#include <glog/logging.h>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>

#ifdef MKLDNN_SUPPORTED
#include "caffe/layers/mkldnn_layers.hpp"
#endif

static const char* supportedEngines[] = {"CAFFE","CUDNN","MKL2017","MKLDNN"};
class EngineParser
{
 public:
   EngineParser(const std::string subEngineString) {
     parse(subEngineString.c_str());
       
     // Check for wrong engine name
     validateEngine();
   }
   const char *getEngineName() const
   {
     return engineName.c_str();
   }
   
   const bool isEngine(const char* name) const
   {
       return (engineName == name);
   }
   
   unsigned getNumberOfSubEngines() const
   {
       return subEngines.size();
   }

#ifdef MKLDNN_SUPPORTED
        engine &getSubEngine(unsigned engineIndex) const
        {
            const char *engineName = subEngines[engineIndex].c_str();

            if(!stricmp(engineName, "CPU"))
                return &CpuEngine::Instance().get_engine();

            if(!stricmp(engineName, "FPGA"))
                return &CpuEngine::Instance().get_engine(); // TODO: change

            LOG(FATAL) << "EngineParser: Unknown subengine";
        }
#endif


    private:

        std::string engineName;
        std::vector<std::string> subEngines;
        
        bool parse(const char *subEngineString)
        {
            // Initialize containers
            engineName.clear();
            subEngines.clear();
            subEngines.reserve(4);

            // Ignore whitespaces
            subEngineString = parseWhitespaces(subEngineString);

            // Extract engine identifier. It can be empty at this point
            const char *beginOfIdentifier = subEngineString;
            subEngineString = parseIdentifier(subEngineString);
            engineName.assign(beginOfIdentifier, subEngineString - beginOfIdentifier);

            // Ignore whitespaces
            subEngineString = parseWhitespaces(subEngineString);

            // String termination is allowed at this place
            if(!*subEngineString)
                return true;

            // Otherwise colon must be specified and engine identifier cannot be empty string
            if(!engineName.length() ||  (*subEngineString != ':'))
                LOG(FATAL) << "Wrong engine specification";

            // Process sub engines
            subEngineString++;
            while(true) {

                // Ignore separators
                subEngineString = parseSeparators(subEngineString);

                // String termination is allowed at this place
                if(!*subEngineString)
                    return true;

                // Extract sub engine identifier
                const char *beginOfIdentifier = subEngineString;
                subEngineString = parseIdentifier(subEngineString);

                // Identifier can not be empty nor contain invalid characters
                if(beginOfIdentifier == subEngineString)
                    return false;

                // Collect all valid sub engine names
                std::string subEngineName;
                subEngineName.assign(beginOfIdentifier, subEngineString - beginOfIdentifier);
                subEngines.push_back(subEngineName);
            }            
        }
        
        void validateEngine() {
            if(subEngines.size() > 0 && engineName != "MKLDNN")
              LOG(FATAL) << "Engine " << engineName << " does not support subengines";
#ifndef USE_CUDNN
            if (engineName == "CUDNN")
                LOG(FATAL) << "Support for CUDNN is not enabled";
#endif
#ifndef MKL2017_SUPPORTED
            if (engineName == "MKL2017")
                LOG(FATAL) << "Support for MKL2017 is not enabled";
#endif
#ifndef MKLDNN_SUPPORTED
            if (engineName == "MKLDNN")
                LOG(FATAL) << "Support for MKLDNN is not enabled";
#endif
            for (unsigned i = 0;
                 i < sizeof(supportedEngines)/sizeof(supportedEngines[0]); i++ )
                if (supportedEngines[i] == engineName)
                    return;
            LOG(FATAL) << "Unknown engine: " << engineName;
        }

        const char *parseWhitespaces(const char *subEngineString) const
        {
            while(isspace(*subEngineString))
                subEngineString++;

            return subEngineString;
        }

        const char *parseSeparators(const char *subEngineString) const
        {
            while(isspace(*subEngineString) || (*subEngineString == ',') || (*subEngineString == ';'))
                subEngineString++;

            return subEngineString;
        }

        const char *parseIdentifier(const char *subEngineString) const
        {
            if(!isalpha(*subEngineString) && (*subEngineString != '_'))
                return subEngineString;

            do subEngineString++;
            while(isalnum(*subEngineString) || (*subEngineString == '_'));

            return subEngineString;
        }
};

#endif
