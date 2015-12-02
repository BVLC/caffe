node-elementtree
====================

node-elementtree is a [Node.js](http://nodejs.org) XML parser and serializer based upon the [Python ElementTree v1.3](http://effbot.org/zone/element-index.htm) module.

Installation
====================

    $ npm install elementtree
    
Using the library
====================

For the usage refer to the Python ElementTree library documentation - [http://effbot.org/zone/element-index.htm#usage](http://effbot.org/zone/element-index.htm#usage).

Supported XPath expressions in `find`, `findall` and `findtext` methods are listed on [http://effbot.org/zone/element-xpath.htm](http://effbot.org/zone/element-xpath.htm).

Example 1 – Creating An XML Document
====================

This example shows how to build a valid XML document that can be published to
Atom Hopper. Atom Hopper is used internally as a bridge from products all the
way to collecting revenue, called “Usage.”  MaaS and other products send similar
events to it every time user performs an action on a resource
(e.g. creates,updates or deletes). Below is an example of leveraging the API
to create a new XML document.

```javascript
var et = require('elementtree');
var XML = et.XML;
var ElementTree = et.ElementTree;
var element = et.Element;
var subElement = et.SubElement;

var date, root, tenantId, serviceName, eventType, usageId, dataCenter, region,
checks, resourceId, category, startTime, resourceName, etree, xml;

date = new Date();

root = element('entry');
root.set('xmlns', 'http://www.w3.org/2005/Atom');

tenantId = subElement(root, 'TenantId');
tenantId.text = '12345';

serviceName = subElement(root, 'ServiceName');
serviceName.text = 'MaaS';

resourceId = subElement(root, 'ResourceID');
resourceId.text = 'enAAAA';

usageId = subElement(root, 'UsageID');
usageId.text = '550e8400-e29b-41d4-a716-446655440000';

eventType = subElement(root, 'EventType');
eventType.text = 'create';

category = subElement(root, 'category');
category.set('term', 'monitoring.entity.create');

dataCenter = subElement(root, 'DataCenter');
dataCenter.text = 'global';

region = subElement(root, 'Region');
region.text = 'global';

startTime = subElement(root, 'StartTime');
startTime.text = date;

resourceName = subElement(root, 'ResourceName');
resourceName.text = 'entity';

etree = new ElementTree(root);
xml = etree.write({'xml_declaration': false});
console.log(xml);
```

As you can see, both et.Element and et.SubElement are factory methods which
return a new instance of Element and SubElement class, respectively.
When you create a new element (tag) you can use set method to set an attribute.
To set the tag value, assign a value to the .text attribute.

This example would output a document that looks like this:

```xml
<entry xmlns="http://www.w3.org/2005/Atom">
  <TenantId>12345</TenantId>
  <ServiceName>MaaS</ServiceName>
  <ResourceID>enAAAA</ResourceID>
  <UsageID>550e8400-e29b-41d4-a716-446655440000</UsageID>
  <EventType>create</EventType>
  <category term="monitoring.entity.create"/>
  <DataCenter>global</DataCenter>
  <Region>global</Region>
  <StartTime>Sun Apr 29 2012 16:37:32 GMT-0700 (PDT)</StartTime>
  <ResourceName>entity</ResourceName>
</entry>
```

Example 2 – Parsing An XML Document
====================

This example shows how to parse an XML document and use simple XPath selectors.
For demonstration purposes, we will use the XML document located at
https://gist.github.com/2554343.

Behind the scenes, node-elementtree uses Isaac’s sax library for parsing XML,
but the library has a concept of “parsers,” which means it’s pretty simple to
add support for a different parser.

```javascript
var fs = require('fs');

var et = require('elementtree');

var XML = et.XML;
var ElementTree = et.ElementTree;
var element = et.Element;
var subElement = et.SubElement;

var data, etree;

data = fs.readFileSync('document.xml').toString();
etree = et.parse(data);

console.log(etree.findall('./entry/TenantId').length); // 2
console.log(etree.findtext('./entry/ServiceName')); // MaaS
console.log(etree.findall('./entry/category')[0].get('term')); // monitoring.entity.create
console.log(etree.findall('*/category/[@term="monitoring.entity.update"]').length); // 1
```

Build status
====================

[![Build Status](https://secure.travis-ci.org/racker/node-elementtree.png)](http://travis-ci.org/racker/node-elementtree)


License
====================

node-elementtree is distributed under the [Apache license](http://www.apache.org/licenses/LICENSE-2.0.html).
