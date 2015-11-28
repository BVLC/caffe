##### Instructions
1. Clone `ember-cli` into a folder under `C:/Users` in Windows or `/Users` in Mac
1. Install Docker on [Windows](https://github.com/boot2docker/windows-installer/releases/latest) or [Mac](https://github.com/boot2docker/osx-installer/releases/latest)
1. Start Boot2Docker
1. `cd` into your cloned `ember-cli` folder
1. Run `cd dev/linux`
1. Run `docker build -t ember-cli .`
1. Run `docker run -ti --rm=true ember-cli`
1. Run `cd ~/ember-cli`
1. Run `npm run-script test-all`
1. When you're done, run `exit`

##### Cleanup
1. Run `docker rmi ember-cli`
