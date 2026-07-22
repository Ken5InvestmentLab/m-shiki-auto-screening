const GITHUB_OWNER = 'Ken5InvestmentLab';
const GITHUB_REPO = 'm-shiki-auto-screening';
const GITHUB_WORKFLOW_FILE = 'screen.yml';
const GITHUB_REF = 'main';
const TIME_ZONE = 'Asia/Tokyo';
const DAILY_TRIGGER_FUNCTION_NAME = 'runDailyMShikiScreening';
const RETRY_TRIGGER_FUNCTION_NAME = 'retryMShikiScreening';
const DISPATCH_HOUR_JST = 16;
const DISPATCH_MINUTE_JST = 0;
const NATIONAL_HOLIDAY_CSV_URL = 'https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv';
const NATIONAL_HOLIDAY_CACHE_KEY = 'national-holidays-csv-v1';
const NATIONAL_HOLIDAY_CACHE_SECONDS = 21600;

function runDailyMShikiScreening() {
  dispatchMShikiScreeningIfReady_();
}

function retryMShikiScreening() {
  deleteTriggers_(RETRY_TRIGGER_FUNCTION_NAME);
  dispatchMShikiScreeningIfReady_();
}

function triggerMShikiScreening() {
  dispatchMShikiScreeningIfReady_();
}

function dispatchMShikiScreeningIfReady_() {
  const now = new Date();
  const closureReason = bankClosureReason_(now);
  if (closureReason) {
    console.log(`Skip closed-day run: ${closureReason}.`);
    return;
  }

  if (isBeforeDispatchTime_(now)) {
    scheduleRetry_(millisecondsUntilDispatch_(now));
    return;
  }

  const token = PropertiesService.getScriptProperties().getProperty('GITHUB_TOKEN');
  if (!token) {
    throw new Error('Missing Script Property: GITHUB_TOKEN');
  }

  const url = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/workflows/${encodeURIComponent(GITHUB_WORKFLOW_FILE)}/dispatches`;
  const payload = {
    ref: GITHUB_REF,
    inputs: {
      dry_run: 'false',
      no_wait: 'true',
      allow_stale_data: 'false',
      as_of_date: '',
      run_at_jst: '',
    },
  };

  const response = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true,
  });

  if (response.getResponseCode() !== 204) {
    throw new Error(`GitHub workflow dispatch failed: ${response.getResponseCode()} ${response.getContentText()}`);
  }

  console.log('M-shiki screening workflow dispatched.');
}

function bankClosureReason_(date) {
  const dayOfWeek = Number(Utilities.formatDate(date, TIME_ZONE, 'u'));
  const monthDay = Utilities.formatDate(date, TIME_ZONE, 'MM-dd');
  const dateKey = Utilities.formatDate(date, TIME_ZONE, 'yyyy/M/d');

  if (dayOfWeek >= 6) {
    return 'weekend';
  }
  if (['12-31', '01-01', '01-02', '01-03'].includes(monthDay)) {
    return 'bank year-end/new-year holiday';
  }
  if (nationalHolidayDateKeys_(dateKey.split('/')[0]).has(dateKey)) {
    return 'Japanese national holiday';
  }
  return '';
}

function nationalHolidayDateKeys_(requiredYear) {
  const cache = CacheService.getScriptCache();
  let csvText = cache.get(NATIONAL_HOLIDAY_CACHE_KEY);

  if (!csvText) {
    const response = UrlFetchApp.fetch(NATIONAL_HOLIDAY_CSV_URL, {
      followRedirects: true,
      muteHttpExceptions: true,
    });
    if (response.getResponseCode() !== 200) {
      throw new Error(`National holiday CSV fetch failed: ${response.getResponseCode()}`);
    }
    csvText = response.getContentText('Shift_JIS');
    cache.put(NATIONAL_HOLIDAY_CACHE_KEY, csvText, NATIONAL_HOLIDAY_CACHE_SECONDS);
  }

  const rows = Utilities.parseCsv(csvText);
  const dateKeys = rows
    .slice(1)
    .map(row => String(row[0] || '').trim())
    .filter(Boolean);
  if (!dateKeys.some(dateKey => dateKey.startsWith(`${requiredYear}/`))) {
    throw new Error(`National holiday CSV has no entries for ${requiredYear}; workflow dispatch stopped.`);
  }
  return new Set(dateKeys);
}

function setupDailyTrigger() {
  deleteDailyTrigger();
  ScriptApp.newTrigger(DAILY_TRIGGER_FUNCTION_NAME)
    .timeBased()
    .everyDays(1)
    .atHour(DISPATCH_HOUR_JST)
    .nearMinute(DISPATCH_MINUTE_JST)
    .inTimezone(TIME_ZONE)
    .create();
}

function deleteDailyTrigger() {
  deleteTriggers_(DAILY_TRIGGER_FUNCTION_NAME);
}

function deleteRetryTrigger() {
  deleteTriggers_(RETRY_TRIGGER_FUNCTION_NAME);
}

function deleteTriggers_(handlerFunction) {
  ScriptApp.getProjectTriggers()
    .filter(trigger => trigger.getHandlerFunction() === handlerFunction)
    .forEach(trigger => ScriptApp.deleteTrigger(trigger));
}

function isBeforeDispatchTime_(date) {
  const hour = Number(Utilities.formatDate(date, TIME_ZONE, 'H'));
  const minute = Number(Utilities.formatDate(date, TIME_ZONE, 'm'));
  return hour < DISPATCH_HOUR_JST || (hour === DISPATCH_HOUR_JST && minute < DISPATCH_MINUTE_JST);
}

function millisecondsUntilDispatch_(date) {
  const hour = Number(Utilities.formatDate(date, TIME_ZONE, 'H'));
  const minute = Number(Utilities.formatDate(date, TIME_ZONE, 'm'));
  const second = Number(Utilities.formatDate(date, TIME_ZONE, 's'));
  const targetSeconds = DISPATCH_HOUR_JST * 3600 + DISPATCH_MINUTE_JST * 60;
  const currentSeconds = hour * 3600 + minute * 60 + second;
  return Math.max((targetSeconds - currentSeconds) * 1000 + 15000, 60000);
}

function scheduleRetry_(delayMilliseconds) {
  deleteTriggers_(RETRY_TRIGGER_FUNCTION_NAME);
  ScriptApp.newTrigger(RETRY_TRIGGER_FUNCTION_NAME)
    .timeBased()
    .after(delayMilliseconds)
    .create();
  console.log(`Before dispatch time. Scheduled retry in ${Math.round(delayMilliseconds / 1000)}s.`);
}
