const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const { spawn } = require('child_process');

let statusBarItem;
let consumerPanel = null;
let providerPanel = null;
let providerProcess = null;
let pollInterval = null;

function requestApi(urlStr, options = {}, postData = null) {
    return new Promise((resolve, reject) => {
        const urlObj = new URL(urlStr);
        const protocol = urlObj.protocol === 'https:' ? https : http;
        
        const req = protocol.request(urlStr, options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    try { resolve(JSON.parse(data)); } catch (e) { resolve(data); }
                } else {
                    reject(new Error(`Status ${res.statusCode}: ${data}`));
                }
            });
        });

        req.on('error', reject);
        
        if (postData) {
            req.write(typeof postData === 'string' ? postData : JSON.stringify(postData));
        }
        req.end();
    });
}

function activate(context) {
    // Status Bar Item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    context.subscriptions.push(statusBarItem);
    updateStatusBar();

    // Listen for config changes
    context.subscriptions.push(vscode.workspace.onDidChangeConfiguration(e => {
        if (e.affectsConfiguration('gpushare.role')) {
            updateStatusBar();
        }
    }));

    // Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('gpushare.openConsumer', () => {
            openConsumerPanel(context);
        }),
        vscode.commands.registerCommand('gpushare.openProvider', () => {
            openProviderPanel(context);
        }),
        vscode.commands.registerCommand('gpushare.configure', async () => {
            const config = vscode.workspace.getConfiguration('gpushare');
            const url = await vscode.window.showInputBox({
                prompt: 'Enter Coordinator URL',
                value: config.get('coordinatorUrl')
            });
            if (url !== undefined) {
                await config.update('coordinatorUrl', url, true);
                vscode.window.showInformationMessage('GPUShare coordinator URL updated.');
            }
        })
    );
    
    startPolling();
}

function updateStatusBar() {
    const config = vscode.workspace.getConfiguration('gpushare');
    const role = config.get('role');
    
    if (role === 'provider') {
        statusBarItem.text = '$(server-process) GPUShare: Provider';
        statusBarItem.command = 'gpushare.openProvider';
    } else {
        statusBarItem.text = '$(circuit-board) GPUShare: Consumer';
        statusBarItem.command = 'gpushare.openConsumer';
    }
    statusBarItem.show();
}

function getWebviewContent(context, viewName) {
    const filePath = path.join(context.extensionPath, 'webviews', `${viewName}.html`);
    return fs.readFileSync(filePath, 'utf8');
}

function openConsumerPanel(context) {
    if (consumerPanel) {
        consumerPanel.reveal(vscode.ViewColumn.One);
        return;
    }

    consumerPanel = vscode.window.createWebviewPanel(
        'gpushareConsumer',
        'GPUShare Consumer',
        vscode.ViewColumn.One,
        { enableScripts: true }
    );

    consumerPanel.webview.html = getWebviewContent(context, 'consumer');
    setupMessagePassing(consumerPanel);

    consumerPanel.onDidDispose(() => {
        consumerPanel = null;
    }, null, context.subscriptions);
}

function openProviderPanel(context) {
    if (providerPanel) {
        providerPanel.reveal(vscode.ViewColumn.One);
        return;
    }

    providerPanel = vscode.window.createWebviewPanel(
        'gpushareProvider',
        'GPUShare Provider',
        vscode.ViewColumn.One,
        { enableScripts: true }
    );

    providerPanel.webview.html = getWebviewContent(context, 'provider');
    setupMessagePassing(providerPanel);

    providerPanel.onDidDispose(() => {
        providerPanel = null;
    }, null, context.subscriptions);
}

function setupMessagePassing(panel) {
    panel.webview.onDidReceiveMessage(
        async message => {
            const config = vscode.workspace.getConfiguration('gpushare');
            const coordinatorUrl = config.get('coordinatorUrl');
            
            try {
                switch (message.type) {
                    case 'submitJob':
                        const res = await requestApi(`${coordinatorUrl}/api/jobs/submit`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' }
                        }, message.data);
                        panel.webview.postMessage({ type: 'jobSubmitted', data: res });
                        break;
                    case 'getNodes':
                        const nodes = await requestApi(`${coordinatorUrl}/api/nodes`);
                        panel.webview.postMessage({ type: 'nodesUpdated', data: nodes });
                        break;
                    case 'getJobStatus':
                        if (message.data && message.data.job_id) {
                            const job = await requestApi(`${coordinatorUrl}/api/jobs/${message.data.job_id}`);
                            panel.webview.postMessage({ type: 'jobStatusUpdated', data: job });
                        }
                        break;
                    case 'getBalance':
                        const balance = await requestApi(`${coordinatorUrl}/api/billing/balance`);
                        panel.webview.postMessage({ type: 'balanceUpdated', data: balance });
                        break;
                    case 'startProvider':
                        if (!providerProcess) {
                            const agentPath = 'C:\\Projects\\GPU Optimization\\provider_agent\\agent.py';
                            
                            providerProcess = spawn('python', [agentPath, '--coordinator', coordinatorUrl], {
                                cwd: 'C:\\Projects\\GPU Optimization\\provider_agent'
                            });
                            
                            providerProcess.stdout.on('data', (data) => {
                                if (providerPanel) providerPanel.webview.postMessage({ type: 'providerLog', data: data.toString() });
                            });
                            
                            providerProcess.stderr.on('data', (data) => {
                                if (providerPanel) providerPanel.webview.postMessage({ type: 'providerLog', data: 'ERROR: ' + data.toString() });
                            });
                            
                            providerProcess.on('close', (code) => {
                                if (providerPanel) providerPanel.webview.postMessage({ type: 'providerLog', data: `Process exited with code ${code}` });
                                providerProcess = null;
                            });
                        }
                        break;
                    case 'stopProvider':
                        if (providerProcess) {
                            providerProcess.kill();
                            providerProcess = null;
                            if (providerPanel) providerPanel.webview.postMessage({ type: 'providerLog', data: 'Provider agent stopped.' });
                        }
                        break;
                }
            } catch (err) {
                vscode.window.showErrorMessage('GPUShare API Error: ' + err.message);
            }
        },
        undefined,
        [] // context.subscriptions not available here without passing
    );
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
        const config = vscode.workspace.getConfiguration('gpushare');
        const coordinatorUrl = config.get('coordinatorUrl');
        
        try {
            const status = await requestApi(`${coordinatorUrl}/api/status`).catch(() => ({ status: 'offline' }));
            if (consumerPanel) consumerPanel.webview.postMessage({ type: 'statusUpdate', data: status });
            if (providerPanel) providerPanel.webview.postMessage({ type: 'statusUpdate', data: status });
            
            const nodes = await requestApi(`${coordinatorUrl}/api/nodes`).catch(() => []);
            if (providerPanel) providerPanel.webview.postMessage({ type: 'nodesUpdated', data: nodes });
        } catch (e) {
            // silent fail for polling
        }
    }, 5000);
}

function deactivate() {
    if (pollInterval) clearInterval(pollInterval);
    if (providerProcess) providerProcess.kill();
}

module.exports = {
    activate,
    deactivate
}
