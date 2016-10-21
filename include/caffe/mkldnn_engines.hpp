#ifndef CAFFE_MKLDNN_ENGINES_HPP_
#define CAFFE_MKLDNN_ENGINES_HPP_

#include <vector>
#include "caffe/layers/mkldnn_layers.hpp"

class EngineSequenceParser
{
    public:

        void parse(const char *engineSequence)
        {
            clear();

            parseSequence(engineSequence);

            useDefaultEngineForEmptySequences();
        }

        void clear()
        {
            engines.clear();
            engines.reserve(8);
        }

        unsigned getNumberOfEngines() const
        {
            return engines.size();
        }

        engine &getEngine(unsigned engineIndex)
        {
            return engines[engineIndex];
        }


    private:

        std::vector<engine> engines;

        void parseSequence(std::string engineSequenceCopy)
        {
            static const char *delimiters = "\t ,;#";

            char *context;
            const char *engineName = strtok_r(&engineSequenceCopy[0], delimiters, &context);

            while(engineName) {
                addEngineByName(engineName);
                engineName = strtok_r(NULL, delimiters, &context);
            }
        }

        void useDefaultEngineForEmptySequences()
        {
            if(engines.empty())
                addEngineByName("CPU");
        }

        void addEngineByName(const char *engineName)
        {
            engine *newEngine = getEngineByName(engineName);

            if(newEngine)
                engines.push_back(*newEngine);

            else
                handleUnsupportedEngine(engineName);
        }

        engine *getEngineByName(const char *engineName)
        {
            if(!stricmp(engineName, "CPU"))
                return &CpuEngine::Instance().get_engine();

            if(!stricmp(engineName, "FPGA"))
                return &CpuEngine::Instance().get_engine();

            return NULL;
        }

        void handleUnsupportedEngine(const char *engineName) const
        {
            CHECK(0) << "Unknown engine specified: '" << current_engine << "'";
        }
};

#endif
