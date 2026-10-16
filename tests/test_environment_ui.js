// A completed job can briefly retain its process lock. Keep polling until the
// lock is released, then refresh dataset readiness exactly once.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

async function main() {
  const scheduled = [], requests = [], notices = [];
  const context = vm.createContext({
    state: {view: 'reports', selectedDatasets: []},
    setTimeout: callback => { scheduled.push(callback); return scheduled.length; },
    request: async route => {
      requests.push(route);
      return route === '/api/datasets' ? [{id: 'mbpp_gen', available: true}]
        : {status: 'completed', busy: false, code: {ready: true}};
    },
    showToast: message => notices.push(message),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../environment_ui.js'), 'utf8'), context);
  vm.runInContext("environmentView.data={status:'completed',busy:true};pollEnvironment()", context);
  assert.equal(scheduled.length, 1);
  assert.equal(vm.runInContext('environmentWorking()', context), true);
  await scheduled[0]();
  assert.equal(vm.runInContext('environmentWorking()', context), false);
  assert.deepEqual(requests, ['/api/environment/status', '/api/datasets']);
  assert.equal(scheduled.length, 1);
  assert.equal(notices.length, 1);
  assert.equal(context.state.datasets[0].id, 'mbpp_gen');
  console.log('Environment UI lock-release polling passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
