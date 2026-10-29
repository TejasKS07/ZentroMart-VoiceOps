'use strict';

const $ = (id) => document.getElementById(id);
const form = $('chat-form');
const input = $('message');
const messages = $('messages');
let busy = false;
let connected = false;
let turn = 0;
let waitingFor = [];
let configuration = null;

const descriptions = {
  input_not_understood: ['A little more detail', 'The issue is unclear. Try one of the examples in the reply.'],
  invalid_order_id: ['Check the order ID', 'The ID was rejected before lookup. An order ID must have exactly 8 digits.'],
  invalid_verification_format: ['Check the verification value', 'The format was not accepted. No identity attempt was used.'],
  order_goal_ambiguous: ['Choose your next step', 'The assistant needs to know whether you are placing or checking an order.'],
  order_placement_unavailable: ['Ordering is unavailable here', 'This workspace can answer support questions and create tickets.'],
  order_id_required: ['Order ID needed', 'Share the 8-digit order ID from My Orders to continue.'],
  verification_required: ['Verification needed', 'Private order details stay hidden until one verification value matches.'],
  verification_failed: ['Verification did not match', 'One retry is allowed. Follow the instructions in the reply.'],
  verification_locked: ['Order details are protected', 'Verification was unsuccessful. General policy help is still available.'],
  order_verified: ['Identity verified', 'The order lookup succeeded for this conversation.'],
  lookup_failed: ['Lookup unavailable', 'The order tool failed. This does not count as an incorrect verification attempt.'],
  ticket_created: ['Ticket created', 'The support ticket was successfully created for a human colleague.'],
  ticket_creation_failed: ['Ticket was not created', 'The handover tool failed. Follow the reply to retry or contact support.'],
  grounded_answer: ['Answer ready', 'The reply uses the available policy sources or verified information.'],
  offline_source_extract: ['Policy reference shared', 'Offline mode is showing original source text rather than a generated policy answer.'],
  offline_unsupported: ['Answer unavailable offline', 'The current local rules do not have a grounded answer to this request.'],
  clarification_required: ['One more detail needed', 'A detail is missing. Answer the question in the conversation to continue.'],
  request_refused: ['A limit applies', 'This request cannot be completed as asked. The reply explains your options.'],
};
const fieldNames = {intent: 'Your support issue', order_id: '8-digit order ID', order_goal: 'New or existing order', verification: 'One verification value', item: 'Which item', refund_method: 'Refund method', public_pincode: 'Delivery pincode', slot_start: 'Delivery slot start', current_time: 'Current local time', policy_detail: 'A policy detail', shopping_question: 'Your shopping question'};
const topicNames = {order: 'Order support', new_order: 'New order', tracking: 'Order tracking', refund: 'Refund', return: 'Return', cancel: 'Cancellation', coupon: 'Coupons', serviceability: 'Delivery availability', delivery_absence: 'Delivery attempts', express_credit: 'Express fee credit', compensation: 'Compensation', damage: 'Damaged item', safety: 'Product safety', human: 'Human support', legal: 'Escalation', unauthorized: 'Account security', membership: 'Plus membership', warranty: 'Warranty', repeat_contact: 'Repeat contact', unrecognized: 'Not identified yet'};

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#i-' + name);
  svg.setAttribute('aria-hidden', 'true');
  svg.append(use);
  return svg;
}

async function api(path, payload) {
  let response;
  try {
    response = await fetch(path, {method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  } catch (_) {
    const error = new Error('The local workspace could not be reached. Check the server, then try again.');
    error.code = 'connection_failed';
    throw error;
  }
  let data;
  try { data = await response.json(); }
  catch (_) { throw new Error('The workspace returned an unreadable response. Please try again.'); }
  if (!response.ok) {
    const error = new Error(data.error?.message || 'The request could not be completed.');
    error.code = data.error?.code;
    throw error;
  }
  return data;
}

function setBusy(value) {
  busy = value;
  input.disabled = value || !connected || Boolean(recorder);
  $('send-button').disabled = value || !connected || !input.value.trim() || Boolean(recorder);
  $('new-conversation').disabled = value || Boolean(recorder);
  document.querySelectorAll('[data-prompt]').forEach((button) => { button.disabled = value || !connected || Boolean(recorder); });
  $('ready-label').textContent = !connected ? 'Disconnected' : value ? 'Working' : 'Ready';
  form.setAttribute('aria-busy', String(value));
  if (typeof updateVoiceControls === 'function') updateVoiceControls();
}

function showConnectionError(message) {
  $('connection-error-text').textContent = message;
  $('connection-error').hidden = false;
}

function setMode(data) {
  configuration = data;
  const live = data.generation_enabled;
  $('sidebar-mode').textContent = live ? 'Policy generation enabled' : 'Offline support is ready';
  $('mode-title').textContent = live ? 'Live policy answers enabled' : 'Working in offline mode';
  $('mode-description').textContent = live ? 'Policy questions can use the configured AI model. Order tools stay local.' : 'Local rules and order tools are available. Gemini generation is disabled.';
}

function resetDetails() {
  turn = 0;
  waitingFor = [];
  $('turn-count').textContent = '—';
  $('result-label').textContent = 'Ready when you are';
  $('result-description').textContent = 'Send a message to see what was understood and what happens next.';
  $('result-banner').className = 'result-banner';
  $('topic-value').textContent = 'Not set';
  $('waiting-value').textContent = 'Your message';
  $('identity-value').textContent = 'Not verified';
  $('source-count').textContent = '0';
  $('technical-output').textContent = 'No turn diagnostics yet.';
  empty($('activity'), 'No tools used yet.');
  empty($('sources'), 'Policy references appear here when they support an answer.');
}

function empty(container, text) {
  const p = document.createElement('p');
  p.className = 'empty-detail';
  p.textContent = text;
  container.replaceChildren(p);
}

async function connect(forceNew = false) {
  if (busy) return;
  setBusy(true);
  try {
    if (!forceNew) {
      const history = await refreshHistory();
      if (history.length) { await openHistory(history[0].id, true); return; }
    }
    const data = await api('/api/session', {});
    activeConversation = data.conversation_id;
    connected = true;
    setMode(data);
    audioUrls.forEach(url => URL.revokeObjectURL(url));
    audioUrls.length = 0;
    messages.replaceChildren();
    $('welcome').hidden = false;
    $('connection-error').hidden = true;
    input.value = '';
    resizeInput();
    resetDetails();
    await refreshHistory();
  } catch (error) {
    connected = false;
    $('sidebar-mode').textContent = 'Connection unavailable';
    showConnectionError(error.message);
  } finally {
    setBusy(false);
  }
}

function scrollToLatest() {
  const body = $('conversation-body');
  body.scrollTop = body.scrollHeight;
}

function addMessage(role, text, pending = false, timestamp = null) {
  const article = document.createElement('article');
  article.className = 'message ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.setAttribute('aria-hidden', 'true');
  if (role === 'assistant') avatar.append(icon('chat'));
  else avatar.textContent = 'YOU';
  const content = document.createElement('div');
  content.className = 'message-content';
  const label = document.createElement('div');
  label.className = 'message-label';
  const name = document.createElement('span');
  name.textContent = role === 'assistant' ? 'ZentroMart assistant' : 'You';
  const time = document.createElement('time');
  const now = timestamp ? new Date(timestamp) : new Date();
  time.dateTime = now.toISOString();
  time.textContent = new Intl.DateTimeFormat(undefined, {hour: 'numeric', minute: '2-digit'}).format(now);
  label.append(name, time);
  const paragraph = document.createElement('p');
  paragraph.className = 'message-text';
  if (pending) {
    paragraph.classList.add('pending-dots');
    paragraph.setAttribute('aria-label', 'Assistant is working');
    for (let i = 0; i < 3; i++) paragraph.append(document.createElement('span'));
  } else paragraph.textContent = text;
  content.append(label, paragraph);
  article.append(avatar, content);
  messages.append(article);
  scrollToLatest();
  return {article, paragraph, content};
}

function updateDetails(response) {
  const diagnostic = response.diagnostics || {};
  waitingFor = diagnostic.waiting_for || [];
  turn += 1;
  $('turn-count').textContent = 'TURN ' + String(turn).padStart(2, '0');
  const [title, description] = descriptions[diagnostic.code] || ['Reply ready', 'The latest response is shown in the conversation.'];
  $('result-label').textContent = title;
  $('result-description').textContent = description;
  $('result-banner').className = 'result-banner' + (diagnostic.status === 'failed' ? ' error' : response.decision === 'clarify' || response.decision === 'refuse' || diagnostic.status === 'limited' ? ' warning' : '');
  $('topic-value').textContent = topicNames[diagnostic.intent] || diagnostic.intent || 'Not set';
  $('waiting-value').textContent = waitingFor.length ? waitingFor.map((name) => fieldNames[name] || name.replaceAll('_', ' ')).join(', ') : 'Nothing needed';
  $('identity-value').textContent = diagnostic.verified ? 'Verified for this order' : 'Not verified';
  $('technical-output').textContent = JSON.stringify(diagnostic, null, 2);
  const events = diagnostic.tools || [];
  if (!events.length) empty($('activity'), 'No tool was needed for this turn.');
  else {
    $('activity').replaceChildren();
    for (const event of events) {
      const row = document.createElement('div');
      row.className = 'activity-item';
      row.append(icon(event.name === 'create_ticket' ? 'chat' : 'box'));
      const text = document.createElement('div');
      text.textContent = event.name === 'lookup_order' ? 'Order lookup' : event.name === 'create_ticket' ? 'Support ticket' : event.name.replaceAll('_', ' ');
      const status = document.createElement('span');
      status.className = 'activity-status';
      status.textContent = {succeeded: 'Succeeded', failed: 'Did not succeed', verification_required: 'Waiting for verification'}[event.status] || event.status;
      text.append(status);
      row.append(text);
      $('activity').append(row);
    }
  }
  const generation = document.createElement('div');
  generation.className = 'activity-item';
  generation.append(icon('info'));
  const label = document.createElement('span');
  label.textContent = diagnostic.llm_used ? 'AI generation used for this turn' : 'Handled locally · no AI generation';
  generation.append(label);
  $('activity').append(generation);
  const sources = response.evidence || [];
  $('source-count').textContent = String(sources.length);
  if (!sources.length) empty($('sources'), 'No policy source was cited in this turn.');
  else {
    $('sources').replaceChildren();
    for (const source of sources) {
      const item = document.createElement('details');
      item.className = 'source-item';
      const heading = document.createElement('summary');
      heading.textContent = source.evidence_id;
      const pre = document.createElement('pre');
      pre.textContent = source.text;
      item.append(heading, pre);
      $('sources').append(item);
    }
  }
}

async function sendMessage(text, retry = null) {
  if (busy || recorder || !connected || !text.trim()) return;
  text = text.trim();
  if (text.length > 2000) return;
  $('welcome').hidden = true;
  const userMessage = retry?.userMessage || addMessage('user', waitingFor.includes('verification') ? 'Verification value sent' : text);
  const requestId = retry?.requestId || crypto.randomUUID();
  retry?.errorNode.remove();
  input.value = '';
  resizeInput();
  setBusy(true);
  const pending = addMessage('assistant', '', true);
  try {
    const data = await api('/api/chat', {message: text, request_id: requestId});
    userMessage.paragraph.textContent = data.display_message;
    pending.article.remove();
    const result = addMessage('assistant', data.response.reply);
    const badge = document.createElement('span');
    badge.className = 'decision-tag ' + data.response.decision;
    badge.textContent = {clarify: 'One more detail', answer: 'Answered', refuse: 'A limit applies', escalate: 'Human support needed'}[data.response.decision] || data.response.decision;
    result.content.append(badge);
    updateDetails(data.response);
    await refreshHistory();
  } catch (error) {
    pending.article.remove();
    const failure = addMessage('assistant', error.message);
    failure.article.classList.add('error');
    $('result-label').textContent = 'Message could not be completed';
    $('result-description').textContent = error.message;
    $('result-banner').className = 'result-banner error';
    $('technical-output').textContent = JSON.stringify({status: 'failed', code: error.code || 'request_failed'}, null, 2);
    if (error.code === 'session_expired') {
      connected = false;
      showConnectionError('This conversation expired. Reconnect to start a fresh conversation.');
    } else {
      const button = document.createElement('button');
      button.className = 'retry-button';
      button.textContent = 'Retry message';
      button.addEventListener('click', () => sendMessage(text, {userMessage, requestId, errorNode: failure.article}));
      failure.content.append(button);
    }
  } finally {
    setBusy(false);
    scrollToLatest();
    if (connected) input.focus({preventScroll: true});
  }
}

function resizeInput() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 132) + 'px';
  $('send-button').disabled = busy || !connected || !input.value.trim();
}

form.addEventListener('submit', (event) => { event.preventDefault(); sendMessage(input.value); });
input.addEventListener('input', resizeInput);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); form.requestSubmit(); }
});
document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => sendMessage(button.dataset.prompt)));
$('new-conversation').addEventListener('click', async () => { await connect(true); if (connected) input.focus(); });
$('reconnect-button').addEventListener('click', () => connect(false));
$('guide-button').addEventListener('click', () => $('guide-dialog').showModal());
$('close-guide').addEventListener('click', () => $('guide-dialog').close());
$('start-chatting').addEventListener('click', () => { $('guide-dialog').close(); input.focus(); });
$('support-nav').addEventListener('click', () => { input.focus(); $('chat-form').scrollIntoView({block: 'nearest'}); });
function toggleDetails(forceOpen = false) {
  const open = forceOpen || !$('inspector').classList.contains('is-open');
  $('inspector').classList.toggle('is-open', open);
  $('details-toggle').setAttribute('aria-expanded', String(open));
  if (open) $('inspector').scrollIntoView({block: 'nearest', behavior: 'smooth'});
}
$('details-toggle').addEventListener('click', () => toggleDetails());
$('status-button').addEventListener('click', () => toggleDetails(true));

// Microphone capture only records audio. Gnani performs all STT and TTS.
let recorder = null;
let recorderContext = null;
let recorderStream = null;
let recorderTimer = null;
let recorderChunks = [];
let recorderSamples = 0;
const audioUrls = [];

function updateVoiceControls() {
  const available = connected && Boolean(configuration?.voice_enabled) && !busy;
  $('record-button').disabled = !available;
  $('upload-button').disabled = !available || Boolean(recorder);
  $('speech-language').disabled = busy || Boolean(recorder);
  if (!recorder) $('voice-status').textContent = configuration?.voice_enabled ? 'Gnani speech · up to 30 seconds' : 'Speech unavailable · check Gnani key and ffmpeg';
}

function base64(bytes) {
  let text = '';
  for (let i=0;i<bytes.length;i+=8192) text += String.fromCharCode(...bytes.subarray(i,i+8192));
  return btoa(text);
}

async function sendAudio(file, language, retry = null) {
  if (busy || !connected || !configuration?.voice_enabled) return;
  if (!file.size || file.size>20*1024*1024) { $('voice-status').textContent='Choose a nonempty audio file under 20 MB.'; return; }
  const extension = '.' + file.name.split('.').pop().toLowerCase();
  if (!['.wav','.mp3','.flac','.ogg','.m4a','.aac','.webm'].includes(extension)) { $('voice-status').textContent='Choose a supported audio file.'; return; }
  $('welcome').hidden = true;
  const userMessage = retry?.userMessage || addMessage('user','Audio message · ' + language);
  const requestId = retry?.requestId || crypto.randomUUID();
  retry?.errorNode.remove();
  setBusy(true);
  updateVoiceControls();
  $('voice-status').textContent = 'Listening with Gnani…';
  const pending = addMessage('assistant','',true);
  try {
    const data = await api('/api/voice',{audio_base64:base64(new Uint8Array(await file.arrayBuffer())),extension,language,request_id:requestId});
    userMessage.paragraph.textContent = data.display_message;
    pending.article.remove();
    const reply = addMessage('assistant',data.response.reply);
    updateDetails(data.response);
    await refreshHistory();
    if (data.reply_audio_base64) {
      const bytes = Uint8Array.from(atob(data.reply_audio_base64),c=>c.charCodeAt(0));
      const url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));
      audioUrls.push(url);
      const player=document.createElement('audio');
      player.className='voice-player'; player.controls=true; player.src=url;
      player.setAttribute('aria-label','Play spoken reply');
      reply.content.append(player);
      // Deliberately require playback control: browsers may block autoplay.
    }
    if (!data.voice.voice_complete) {
      const error=document.createElement('p'); error.className='voice-error';
      error.textContent=data.voice.error?.message || 'This voice turn needs a clearer recording.';
      reply.content.append(error);
      if (data.voice.error?.stage==='tts') {
        const button=document.createElement('button'); button.className='retry-button';button.textContent='Retry spoken reply';
        button.addEventListener('click',()=>sendAudio(file,language,{userMessage,requestId,errorNode:reply.article}));
        reply.content.append(button);
      }
    }
    const technical={...data.response.diagnostics,voice:data.voice};
    $('technical-output').textContent=JSON.stringify(technical,null,2);
    $('voice-status').textContent=data.voice.voice_complete?'Spoken reply ready':'Voice turn incomplete';
  } catch(error) {
    pending.article.remove();
    const failure=addMessage('assistant',error.message); failure.article.classList.add('error');
    const button=document.createElement('button');button.className='retry-button';button.textContent='Retry audio';
    button.addEventListener('click',()=>sendAudio(file,language,{userMessage,requestId,errorNode:failure.article}));
    failure.content.append(button);
    $('voice-status').textContent='Speech failed · text support is still available';
  } finally {
    setBusy(false);updateVoiceControls();scrollToLatest();
  }
}

function encodeWav(chunks, sampleCount, rate) {
  const buffer=new ArrayBuffer(44+sampleCount*2),view=new DataView(buffer);
  const word=(offset,value)=>{for(let i=0;i<value.length;i++)view.setUint8(offset+i,value.charCodeAt(i));};
  word(0,'RIFF');view.setUint32(4,36+sampleCount*2,true);word(8,'WAVE');word(12,'fmt ');
  view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,rate,true);
  view.setUint32(28,rate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);word(36,'data');view.setUint32(40,sampleCount*2,true);
  let offset=44;
  for(const chunk of chunks)for(const value of chunk){const sample=Math.max(-1,Math.min(1,value));view.setInt16(offset,sample<0?sample*32768:sample*32767,true);offset+=2;}
  return new File([buffer],'recording.wav',{type:'audio/wav'});
}

async function stopRecording(send=true) {
  if (!recorder) return;
  clearTimeout(recorderTimer);
  const rate=recorderContext.sampleRate;
  recorder.port.onmessage=null;recorder.disconnect();
  recorderStream.getTracks().forEach(track=>track.stop());
  await recorderContext.close();
  recorder=null;recorderContext=null;recorderStream=null;
  $('record-button').classList.remove('recording');$('record-label').textContent='Record';
  $('new-conversation').disabled=false;
  input.disabled=!connected;
  const file=encodeWav(recorderChunks,recorderSamples,rate);
  recorderChunks=[];recorderSamples=0;
  updateVoiceControls();
  if(send)await sendAudio(file,$('speech-language').value);
}

async function startRecording() {
  if(busy || !connected) return;
  if(recorder){await stopRecording();return;}
  try {
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('Microphone recording is unavailable; upload an audio file instead.');
    recorderStream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true},video:false});
    recorderContext=new AudioContext();
    await recorderContext.audioWorklet.addModule('/recorder-worklet.js');
    recorderChunks=[];recorderSamples=0;
    recorder=new AudioWorkletNode(recorderContext,'voice-recorder');
    recorder.port.onmessage=({data})=>{const remaining=Math.max(0,Math.floor(recorderContext.sampleRate*29.8)-recorderSamples);if(remaining){const chunk=data.slice(0,remaining);recorderChunks.push(chunk);recorderSamples+=chunk.length;}};
    const source=recorderContext.createMediaStreamSource(recorderStream);source.connect(recorder);
    const mute=recorderContext.createGain();mute.gain.value=0;recorder.connect(mute);mute.connect(recorderContext.destination);
    $('record-button').classList.add('recording');$('record-label').textContent='Stop & send';
    setBusy(false);
    $('voice-status').textContent='Recording… click Stop & send';
    $('upload-button').disabled=true;$('speech-language').disabled=true;$('new-conversation').disabled=true;input.disabled=true;
    recorderTimer=setTimeout(()=>stopRecording(),29800);
  } catch(error) {
    recorderStream?.getTracks().forEach(track=>track.stop());
    if(recorderContext)await recorderContext.close();
    recorder=null;recorderContext=null;recorderStream=null;
    $('voice-status').textContent=error.name==='NotAllowedError'?'Microphone permission declined. Upload audio or use text.':error.message;
  }
}

$('record-button').addEventListener('click',startRecording);
$('upload-button').addEventListener('click',()=>$('audio-file').click());
$('audio-file').addEventListener('change',async()=>{const file=$('audio-file').files[0];if(file)await sendAudio(file,$('speech-language').value);$('audio-file').value='';});
window.addEventListener('beforeunload',()=>{recorderStream?.getTracks().forEach(track=>track.stop());audioUrls.forEach(url=>URL.revokeObjectURL(url));});


let activeConversation = null;

async function refreshHistory() {
  const response=await fetch('/api/history',{credentials:'same-origin'});
  if(!response.ok)throw new Error('Chat history could not be loaded.');
  const data=await response.json();
  const list=$('history-list');list.replaceChildren();
  if(!data.conversations.length){const p=document.createElement('p');p.className='history-empty';p.textContent='Your conversations will appear here.';list.append(p);}
  for(const conversation of data.conversations){
    const row=document.createElement('div');row.className='history-row'+(conversation.id===activeConversation?' current':'');
    const button=document.createElement('button');button.className='history-item';
    const title=document.createElement('strong');title.textContent=conversation.title;
    const timestamp=document.createElement('time');timestamp.dateTime=conversation.updated;
    timestamp.textContent=new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}).format(new Date(conversation.updated));
    button.append(title,timestamp);button.addEventListener('click',()=>openHistory(conversation.id));
    const remove=document.createElement('button');remove.className='history-delete';remove.setAttribute('aria-label','Delete '+conversation.title);remove.append(icon('close'));
    remove.addEventListener('click',async()=>{if(busy||recorder)return;try{await api('/api/history/delete',{conversation_id:conversation.id});if(activeConversation===conversation.id)await connect(true);else await refreshHistory();}catch(error){showConnectionError(error.message);}});
    row.append(button,remove);list.append(row);
  }
  return data.conversations;
}

function attachHistoricalAudio(content, encoded) {
  if(!encoded)return;
  const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
  const url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));audioUrls.push(url);
  const player=document.createElement('audio');player.controls=true;player.className='voice-player';player.src=url;player.setAttribute('aria-label','Play saved spoken reply');content.append(player);
}

async function openHistory(conversationId, duringConnect=false) {
  if((busy&&!duringConnect)||recorder)return;
  if(!duringConnect)setBusy(true);
  try{
    const data=await api('/api/history/select',{conversation_id:conversationId});
    activeConversation=conversationId;connected=true;setMode(data);
    audioUrls.forEach(url=>URL.revokeObjectURL(url));audioUrls.length=0;
    messages.replaceChildren();resetDetails();$('welcome').hidden=Boolean(data.messages.length);
    input.value='';resizeInput();
    for(const turn of data.messages){addMessage('user',turn.display_message||'Audio message',false,turn.created_at);const reply=addMessage('assistant',turn.response.reply,false,turn.created_at);attachHistoricalAudio(reply.content,turn.reply_audio_base64);updateDetails(turn.response);}
    if(data.verification_required_on_continue&&data.messages.length){
      $('identity-value').textContent='Verify again to continue';
      $('result-label').textContent='Conversation restored';
      $('result-description').textContent='History is saved. Order verification must be repeated before continuing with private order details.';
      $('result-banner').className='result-banner warning';
    }
    $('connection-error').hidden=true;
    document.querySelector('.sidebar').classList.remove('history-open');
    await refreshHistory();scrollToLatest();
  }catch(error){showConnectionError(error.message);}
  finally{if(!duringConnect)setBusy(false);}
}

$('history-toggle').addEventListener('click',()=>document.querySelector('.sidebar').classList.add('history-open'));
$('close-history').addEventListener('click',()=>document.querySelector('.sidebar').classList.remove('history-open'));

connect();
