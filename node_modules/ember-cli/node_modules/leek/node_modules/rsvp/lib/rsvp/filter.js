import Promise from './promise';
import {
  isFunction
} from './utils';

/**
 `RSVP.filter` is similar to JavaScript's native `filter` method, except that it
  waits for all promises to become fulfilled before running the `filterFn` on
  each item in given to `promises`. `RSVP.filter` returns a promise that will
  become fulfilled with the result of running `filterFn` on the values the
  promises become fulfilled with.

  For example:

  ```javascript

  var promise1 = RSVP.resolve(1);
  var promise2 = RSVP.resolve(2);
  var promise3 = RSVP.resolve(3);

  var promises = [promise1, promise2, promise3];

  var filterFn = function(item){
    return item > 1;
  };

  RSVP.filter(promises, filterFn).then(function(result){
    // result is [ 2, 3 ]
  });
  ```

  If any of the `promises` given to `RSVP.filter` are rejected, the first promise
  that is rejected will be given as an argument to the returned promise's
  rejection handler. For example:

  ```javascript
  var promise1 = RSVP.resolve(1);
  var promise2 = RSVP.reject(new Error('2'));
  var promise3 = RSVP.reject(new Error('3'));
  var promises = [ promise1, promise2, promise3 ];

  var filterFn = function(item){
    return item > 1;
  };

  RSVP.filter(promises, filterFn).then(function(array){
    // Code here never runs because there are rejected promises!
  }, function(reason) {
    // reason.message === '2'
  });
  ```

  `RSVP.filter` will also wait for any promises returned from `filterFn`.
  For instance, you may want to fetch a list of users then return a subset
  of those users based on some asynchronous operation:

  ```javascript

  var alice = { name: 'alice' };
  var bob   = { name: 'bob' };
  var users = [ alice, bob ];

  var promises = users.map(function(user){
    return RSVP.resolve(user);
  });

  var filterFn = function(user){
    // Here, Alice has permissions to create a blog post, but Bob does not.
    return getPrivilegesForUser(user).then(function(privs){
      return privs.can_create_blog_post === true;
    });
  };
  RSVP.filter(promises, filterFn).then(function(users){
    // true, because the server told us only Alice can create a blog post.
    users.length === 1;
    // false, because Alice is the only user present in `users`
    users[0] === bob;
  });
  ```

  @method filter
  @static
  @for RSVP
  @param {Array} promises
  @param {Function} filterFn - function to be called on each resolved value to
  filter the final results.
  @param {String} label optional string describing the promise. Useful for
  tooling.
  @return {Promise}
*/
export default function filter(promises, filterFn, label) {
  return Promise.all(promises, label).then(function(values) {
    if (!isFunction(filterFn)) {
      throw new TypeError("You must pass a function as filter's second argument.");
    }

    var length = values.length;
    var filtered = new Array(length);

    for (var i = 0; i < length; i++) {
      filtered[i] = filterFn(values[i]);
    }

    return Promise.all(filtered, label).then(function(filtered) {
      var results = new Array(length);
      var newLength = 0;

      for (var i = 0; i < length; i++) {
        if (filtered[i]) {
          results[newLength] = values[i];
          newLength++;
        }
      }

      results.length = newLength;

      return results;
    });
  });
}
