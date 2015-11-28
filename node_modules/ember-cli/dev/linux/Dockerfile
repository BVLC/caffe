FROM node:latest

RUN npm i -g npm
RUN npm i -g phantomjs

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/ember-cli/ember-cli.git ~/ember-cli
RUN cd ~/ember-cli && npm i

ENTRYPOINT ["/bin/bash"]
