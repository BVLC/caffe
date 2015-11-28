<grey>You may generate models with as many attrs as you would like to pass. The following attribute types are supported:</grey>
  <yellow><attr-name></yellow>
  <yellow><attr-name></yellow>:array
  <yellow><attr-name></yellow>:boolean
  <yellow><attr-name></yellow>:date
  <yellow><attr-name></yellow>:object
  <yellow><attr-name></yellow>:number
  <yellow><attr-name></yellow>:string
  <yellow><attr-name></yellow>:your-custom-transform
  <yellow><attr-name></yellow>:belongs-to:<yellow><model-name></yellow>
  <yellow><attr-name></yellow>:has-many:<yellow><model-name></yellow>

For instance: <green>\`ember generate model taco filling:belongs-to:protein toppings:has-many:toppings name:string price:number misc\`</green>
would result in the following model:

```js
import DS from 'ember-data';
export default DS.Model.extend({
  filling: DS.belongsTo('protein'),
  toppings: DS.hasMany('topping'),
  name: DS.attr('string'),
  price: DS.attr('number'),
  misc: DS.attr()
});
```
