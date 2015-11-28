/**
 * random table string and table length.
 */
var TABLE = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
    TABLE_LEN = TABLE.length;


/**
 * generate random string from template.
 *
 * replace for placeholder "X" in template.
 * return template if not has placeholder.
 *
 * @param {String} template template string.
 * @throws {TypeError} if template is not a String.
 * @return {String} replaced string.
 */
function generate(template) {
  var match, i, len, result;

  if (typeof template !== 'string') {
    throw new TypeError('template must be a String: ' + template);
  }

  match = template.match(/(X+)[^X]*$/);

  // return template if not has placeholder
  if (match === null) {
    return template;
  }

  // generate random string
  for (result = '', i = 0, len = match[1].length; i < len; ++i) {
    result += TABLE[Math.floor(Math.random() * TABLE_LEN)];
  }

  // concat template and random string
  return template.slice(0, match.index) + result +
      template.slice(match.index + result.length);
}


/**
 * export.
 */
module.exports = {
  generate: generate
};
