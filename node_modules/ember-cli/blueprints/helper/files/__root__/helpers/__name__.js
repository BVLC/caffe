import Ember from 'ember';

export function <%= camelizedModuleName %>(params/*, hash*/) {
  return params;
}

export default Ember.Helper.helper(<%= camelizedModuleName %>);
