const express = require('express');
const app = express();
app.get('/', (req, res) => {
  res.send('Hello from Node.js Express!');
});
app.listen(3000, () => {
  console.log('Node app listening on port 3000');
});
