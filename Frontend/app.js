const API_BASE = 'http://localhost:8000';

const $ = (id) => document.getElementById(id);
const video = $('camera');
const canvas = document.createElement('canvas');
canvas.width = 640; canvas.height = 480;
const ctx = canvas.getContext('2d', {willReadFrequently:false});
let cameraStream = null;
let backendOnline = false;
let activeUser = '';
let activeSession = '';
let latestResponse = null;
let stepUpResponse = null;
let busyInterval = null;
let autoHomeTimeout = null;
let requestInProgress = false;
let attemptLog = [];

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function showView(name, label) {
  document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
  $('view' + name[0].toUpperCase() + name.slice(1)).classList.add('active');
  $('stepLabel').textContent = label || ({home:'SECURE ENTRY',register:'NEW ENROLLMENT',enrollment:'AUTHENTICATOR SETUP',busy:'IDENTITY CHECK',result:'ACCESS DECISION',challenge:'ACTIVE CHALLENGE',otp:'TWO-FACTOR AUTH',message:'SYSTEM MESSAGE'}[name] || 'SECURE ENTRY');
  clearTimeout(autoHomeTimeout);
}
function setGuide(decision) {
  const guide = $('faceGuide');
  guide.classList.remove('guide-green','guide-amber','guide-red');
  const color = {ALLOW:'guide-green',STEP_UP:'guide-amber',DENY:'guide-red',RECAPTURE:''}[decision];
  if (color) guide.classList.add(color);
}
function setHint(text) { $('cameraHint').textContent = text; }

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    $('cameraMessage').innerHTML = '<span class="camera-icon">!</span><span>Camera unavailable in this browser</span>';
    setHint('Use localhost and allow camera access to continue'); return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});
    video.srcObject = cameraStream;
    await video.play();
    $('cameraMessage').hidden = true;
    setHint('Position your face inside the guide');
  } catch (error) {
    $('cameraMessage').innerHTML = '<span class="camera-icon">!</span><span>Camera access is needed to verify</span>';
    setHint('Allow camera access in your browser settings');
  }
}
function cameraReady() { return cameraStream && video.readyState >= 2 && video.videoWidth > 0; }
async function captureBurst(count, intervalMs) {
  if (!cameraReady()) throw new Error('Camera is not ready. Allow camera access, then try again.');
  const frames = [];
  for (let i=0;i<count;i++) {
    ctx.drawImage(video,0,0,640,480);
    frames.push(canvas.toDataURL('image/jpeg',0.8));
    if (i < count-1) await sleep(intervalMs);
    $('progressBar').style.width = `${Math.round(((i+1)/count)*100)}%`;
  }
  return frames;
}
async function api(path, options={}) {
  const controller = new AbortController();
  const timeout = setTimeout(()=>controller.abort(),20000);
  try {
    const response = await fetch(API_BASE+path,{...options,signal:controller.signal,headers:{'Content-Type':'application/json',...(options.headers||{})}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The backend took too long to respond. Please try again.');
    if (error instanceof TypeError) throw new Error('Cannot reach the access server. Check that the backend is running.');
    throw error;
  } finally { clearTimeout(timeout); }
}
async function post(path, body) { return api(path,{method:'POST',body:JSON.stringify(body)}); }

async function pollHealth() {
  try {
    const health = await api('/health');
    backendOnline = !!health.ok;
    $('onlineBadge').classList.toggle('offline',!backendOnline);
    $('onlineBadge').innerHTML = `<span></span><b>${backendOnline?'SYSTEM ONLINE':'OFFLINE'}</b>`;
    $('offlineBanner').hidden = backendOnline;
  } catch (_) {
    backendOnline = false;
    $('onlineBadge').classList.add('offline');
    $('onlineBadge').innerHTML = '<span></span><b>OFFLINE</b>';
    $('offlineBanner').hidden = false;
  }
}
function startBusy(title, text) {
  showView('busy'); $('busyTitle').innerHTML = title; $('busyText').textContent = text;
  $('progressBar').style.width='3%';
  const started=performance.now();
  clearInterval(busyInterval);
  busyInterval=setInterval(()=>{$('stopwatch').innerHTML=`${((performance.now()-started)/1000).toFixed(2)}<span>s</span>`;},35);
}
function stopBusy() { clearInterval(busyInterval); busyInterval=null; }
function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  const n=Number(value); return `${Math.round((n>=0&&n<=1?n*100:n))}%`;
}
function scoreValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  const n=Number(value); return Math.max(0,Math.min(100,Math.round(n>=0&&n<=1?n*100:n)));
}
// "Liveness 87% · Match 76% · 0.6 s" -- the single small line under the message.
function summaryLine(data) {
  const parts=[];
  const live=percent(data.liveness); if(live) parts.push(`Liveness ${live}`);
  const match=percent(data.match); if(match) parts.push(`Match ${match}`);
  if(data.time_ms!==undefined&&data.time_ms!==null) parts.push(`${(data.time_ms/1000).toFixed(1)} s`);
  return parts.join(' · ');
}
function addAttempt(name, decision, ms) {
  attemptLog.unshift({time:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),name:name||'Unknown',decision:decision||'DENY',ms});
  attemptLog=attemptLog.slice(0,10);
  $('attemptLog').innerHTML=attemptLog.map(item=>{
    const cls={ALLOW:'',STEP_UP:'step',DENY:'deny',RECAPTURE:'recap'}[item.decision]||'deny';
    return `<span class="attempt-entry ${cls}"><span>${escapeHtml(item.time)}</span><b>${escapeHtml(item.name)}</b><span class="entry-decision">${escapeHtml(item.decision)}</span><span>${item.ms??'—'}ms</span></span>`;
  }).join('');
}
function escapeHtml(value) { return String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function showResult(data, opts={}) {
  stopBusy(); requestInProgress=false;
  latestResponse=data; $('debugJson').textContent=JSON.stringify(data,null,2);
  const decision=data.decision||'DENY'; setGuide(decision);
  const root=$('viewResult'); root.classList.remove('result-allow','result-step','result-deny','result-recapture');
  root.classList.add(({ALLOW:'result-allow',STEP_UP:'result-step',DENY:'result-deny',RECAPTURE:'result-recapture'})[decision]||'result-deny');
  // The backend owns the wording: one title + one message per outcome. The
  // fallbacks only cover an older backend that does not send them yet.
  const fallback={ALLOW:['✓','Access granted'],STEP_UP:['↗','One more step'],DENY:['×','Access denied'],RECAPTURE:['↻',"Let's try again"]}[decision]||['×','Access denied'];
  $('resultIcon').textContent=fallback[0];
  $('resultTitle').textContent=data.title||fallback[1];
  const message=data.message||(data.reasons||[])[0]||'';
  $('resultMessage').textContent=message; $('resultMessage').hidden=!message;
  $('resultSummary').textContent=summaryLine(data);
  $('resultDetails').open=false;
  $('liveMetric').textContent=percent(data.liveness)??'—'; $('matchMetric').textContent=percent(data.match)??'—';
  $('timeMetric').textContent=data.time_ms!==undefined?`${data.time_ms} ms`:'—';
  $('scheduleReason').hidden=!data.schedule_reason; $('scheduleReason').textContent=data.schedule_reason||'';
  $('reasonChips').innerHTML=(data.reasons||[]).map(reason=>`<span class="reason-chip">${escapeHtml(reason)}</span>`).join('');
  const scores=data.scores||{};
  const signals=[['ANTI-SPOOF',scores.passive],['MOTION',scores.motion],['BLINK',scores.blink],['DEVICE',scores.device],['MATCH',scores.match??data.match]];
  const lat=scores.latency_ms||{};
  const latencyList=Object.entries(lat).map(([k,v])=>`${k.replace('_ms','').replace('blinkmotion','motion')}: ${v}ms`).join(' · ');
  $('latencySummary').textContent=latencyList;
  $('signalRows').innerHTML=signals.map(([label,value])=>{
    let val=scoreValue(value); if(label==='DEVICE'&&value!==null&&value!==undefined) val=value?100:0;
    const display=label==='DEVICE'?(value===null||value===undefined?'—':(value?'DETECTED':'CLEAR')):(val===null?'—':`${val}%`);
    return `<div class="signal-row"><span>${label}</span><div class="signal-track"><i style="width:${val??0}%"></i></div><span class="signal-val">${display}</span><span class="signal-time">${label==='ANTI-SPOOF'?`${lat.passive_ms??'—'}ms`:label==='MOTION'||label==='BLINK'?`${lat.blinkmotion_ms??'—'}ms`:label==='DEVICE'?`${lat.device_ms??'—'}ms`:' '}</span></div>`;
  }).join('');
  showView('result'); $('stepLabel').textContent='ACCESS DECISION';
  const fromOtp=!!opts.fromOtp;
  addAttempt(data.name||activeUser,decision,data.time_ms);
  if(decision!=='STEP_UP'||fromOtp) autoHomeTimeout=setTimeout(goHome,20000);
}
function showMessage(title,text) { stopBusy();requestInProgress=false;$('messageTitle').textContent=title;$('messageText').textContent=text;showView('message'); }
function goHome() {
  stopBusy(); requestInProgress=false; activeSession=''; activeUser='';
  showView('home'); $('stepLabel').textContent='SECURE ENTRY';
  $('demoCode').textContent='······'; $('otpInput').value=''; $('otpError').hidden=true;
  $('faceGuide').classList.remove('guide-green','guide-amber','guide-red'); setHint('Position your face inside the guide');
}
async function beginVerify() {
  if(requestInProgress)return;
  if(!backendOnline){showMessage('Server offline','The access server is unavailable. No entry decision can be made. Please try again when it is online.');return;}
  requestInProgress=true; startBusy('Checking<br><span>liveness.</span>','Keep looking at the camera.');setHint('Hold still while your face is checked');
  try { const frames=await captureBurst(10,100); $('busyText').textContent='Comparing your face and checking liveness.'; const data=await post('/verify',{frames});activeUser=data.name||'';activeSession=data.session_id||'';await continueDecision(data); }
  catch(error){showMessage('Verification unavailable',error.message);}
}
async function continueDecision(data) {
  latestResponse=data; $('debugJson').textContent=JSON.stringify(data,null,2);
  if(data.decision==='STEP_UP') stepUpResponse=data;
  if(data.decision==='STEP_UP'&&data.challenge){activeSession=data.session_id||activeSession;$('challengeText').textContent=data.message||data.challenge.text||'Please complete the challenge';$('countdown').hidden=true;showView('challenge');setGuide('STEP_UP');setHint('Follow the on-screen instruction');requestInProgress=false;return;}
  if(data.totp_required){activeSession=data.session_id||activeSession;if(data.message)$('otpLede').textContent=data.message;showView('otp');setGuide('STEP_UP');setHint('Complete authenticator verification');requestInProgress=false;return;}
  showResult(data);
}
async function beginChallenge() {
  if(requestInProgress)return;
  if(!backendOnline){showMessage('Server offline','The access server is unavailable. Please try again when it is online.');return;}
  requestInProgress=true;$('challengeBtn').disabled=true;$('countdown').hidden=false;
  try {
    for(let n=3;n>=1;n--){$('countdown').textContent=n;await sleep(650);}
    $('countdown').hidden=true;startBusy('Completing<br><span>challenge.</span>','Keep following the instruction.');
    const frames=await captureBurst(20,100);$('busyText').textContent='Checking challenge response.';
    const data=await post('/challenge',{session_id:activeSession,frames});await continueDecision(data);
  } catch(error){showMessage('Challenge unavailable',error.message);} finally {$('challengeBtn').disabled=false;$('countdown').hidden=true;}
}
async function beginEnroll() {
  const name=$('nameInput').value.trim(); $('enrollError').hidden=true;
  if(!name){$('enrollError').textContent='Enter your name to continue.';$('enrollError').hidden=false;return;}
  if(!backendOnline){$('enrollError').textContent='The access server is offline. Try again when it is online.';$('enrollError').hidden=false;return;}
  if(requestInProgress)return;requestInProgress=true;$('enrollBtn').disabled=true;
  try {
    showView('busy');$('busyTitle').innerHTML='Get ready to<br><span>enroll.</span>';$('busyText').textContent='Keep your face inside the guide.';
    for(let n=3;n>=1;n--){$('busyText').textContent=`Capturing in ${n}… look straight at the camera.`;await sleep(650);}
    const frames=await captureBurst(5,200);const data=await post('/enroll',{name,frames});
    $('debugJson').textContent=JSON.stringify(data,null,2);latestResponse=data;
    if(!data.ok) throw new Error(data.reason||'Enrollment could not be completed.');
    activeUser=data.name||name;$('qrImage').src=data.totp_qr;
    $('enrollSetup').hidden=false;$('enrollComplete').hidden=true;$('confirmCodeInput').value='';$('confirmError').hidden=true;
    showView('enrollment');setHint('Scan the QR code, then enter the 6-digit code');$('confirmCodeInput').focus();
  } catch(error){$('enrollError').textContent=error.message;$('enrollError').hidden=false;showView('register');}
  finally{requestInProgress=false;$('enrollBtn').disabled=false;}
}
async function confirmEnroll() {
  const code=$('confirmCodeInput').value.replace(/\D/g,'');$('confirmError').hidden=true;
  if(code.length!==6){$('confirmError').textContent='Enter all 6 digits.';$('confirmError').hidden=false;return;}
  if(!backendOnline){$('confirmError').textContent='The access server is offline. Try again when it is online.';$('confirmError').hidden=false;return;}
  if(requestInProgress)return;requestInProgress=true;$('confirmEnrollBtn').disabled=true;
  try {
    const data=await post('/enroll/confirm',{name:activeUser,code});
    $('debugJson').textContent=JSON.stringify(data,null,2);
    if(data.ok){
      $('enrolledName').textContent=`${activeUser} can now use this kiosk`;
      $('enrollSetup').hidden=true;$('enrollComplete').hidden=false;setHint('Registration complete');
    } else {
      $('confirmError').textContent="Code didn't match — try again";$('confirmError').hidden=false;
    }
  } catch(error){$('confirmError').textContent=error.message;$('confirmError').hidden=false;}
  finally{requestInProgress=false;$('confirmEnrollBtn').disabled=false;}
}
async function submitOtp() {
  const code=$('otpInput').value.replace(/\D/g,'');$('otpError').hidden=true;
  if(code.length!==6){$('otpError').textContent='Enter all 6 digits.';$('otpError').hidden=false;return;}
  if(requestInProgress)return;requestInProgress=true;$('otpBtn').disabled=true;
  try {
    const data=await post('/otp',{session_id:activeSession,code});
    latestResponse=data;$('debugJson').textContent=JSON.stringify(data,null,2);
    showResult({...data,liveness:stepUpResponse?.liveness,match:stepUpResponse?.match,time_ms:stepUpResponse?.time_ms,schedule_reason:stepUpResponse?.schedule_reason,reasons:[data.reason].filter(Boolean),scores:stepUpResponse?.scores},{fromOtp:true});
  } catch(error){$('otpError').textContent=error.message;$('otpError').hidden=false;requestInProgress=false;}
  finally{$('otpBtn').disabled=false;}
}
async function showDemoCode() {
  if(!activeUser){$('demoCode').textContent='Register or verify a user first';return;}
  try {const data=await api(`/dev/totp/${encodeURIComponent(activeUser)}`);$('demoCode').textContent=data.code||data.error||'Unavailable';}
  catch(error){$('demoCode').textContent='Backend offline';}
}
async function setForce(decision) {
  try {const result=await api(`/dev/force?decision=${encodeURIComponent(decision)}`);$('forceStatus').textContent=`Decision override: ${result.forced||'off'}`;await pollHealth();}
  catch(error){$('forceStatus').textContent=error.message;}
}

$('verifyBtn').addEventListener('click',beginVerify);
$('registerBtn').addEventListener('click',()=>{showView('register');$('nameInput').focus();});
$('enrollBtn').addEventListener('click',beginEnroll);
$('nameInput').addEventListener('keydown',e=>{if(e.key==='Enter')beginEnroll();});
$('confirmEnrollBtn').addEventListener('click',confirmEnroll);
$('confirmCodeInput').addEventListener('input',()=>{$('confirmCodeInput').value=$('confirmCodeInput').value.replace(/\D/g,'').slice(0,6);});
$('confirmCodeInput').addEventListener('keydown',e=>{if(e.key==='Enter')confirmEnroll();});
$('challengeBtn').addEventListener('click',beginChallenge);
$('otpBtn').addEventListener('click',submitOtp);
$('otpInput').addEventListener('input',()=>{$('otpInput').value=$('otpInput').value.replace(/\D/g,'').slice(0,6);});
$('otpInput').addEventListener('keydown',e=>{if(e.key==='Enter')submitOtp();});
$('demoCodeBtn').addEventListener('click',showDemoCode);
document.querySelectorAll('[data-home]').forEach(button=>button.addEventListener('click',goHome));
document.querySelectorAll('[data-force]').forEach(button=>button.addEventListener('click',()=>setForce(button.dataset.force)));
$('closeDebug').addEventListener('click',()=>$('debugPanel').hidden=true);
document.addEventListener('keydown',e=>{
  if(e.key.toLowerCase()==='d'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName))$('debugPanel').hidden=!$('debugPanel').hidden;
  if(e.key.toLowerCase()==='f'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName))$('demoMenu').hidden=!$('demoMenu').hidden;
  if(e.key==='Escape'){ $('debugPanel').hidden=true;$('demoMenu').hidden=true; }
});

// Small manual "Done" button on the result card (auto-return is now 20s).
(function addResultDoneButton(){
  const resultView=$('viewResult'); if(!resultView||$('resultDoneBtn'))return;
  const btn=document.createElement('button');
  btn.id='resultDoneBtn'; btn.type='button'; btn.textContent='Done';
  btn.className='btn'; btn.style.cssText='margin-top:10px;padding:6px 20px;font-size:0.85rem;opacity:0.85;';
  btn.addEventListener('click',goHome);
  const anchor=resultView.querySelector('.result-home');
  if(anchor&&anchor.parentNode)anchor.parentNode.appendChild(btn); else resultView.appendChild(btn);
})();

startCamera();pollHealth();setInterval(pollHealth,5000);
