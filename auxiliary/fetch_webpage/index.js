'use strict';

// Original author: https://github.com/SmartHypercube
// Please refer to https://github.com/Sparticuz/chromium to deploy on AWS Lambda

const chromium = require("@sparticuz/chromium");
//const puppeteer = require("puppeteer-core");
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());
const AdblockerPlugin = require('puppeteer-extra-plugin-adblocker');
puppeteer.use(AdblockerPlugin({ blockTrackers: true }));

// https://github.com/puppeteer/puppeteer/issues/305#issuecomment-385145048
async function scroll(page) {
  await page.evaluate(async () => {
    await new Promise((resolve, reject) => {
      function end() {
        // 滚动到顶部再返回
        clearInterval(timer1);
        clearTimeout(timer2);
        window.scrollTo(0, 0);
        resolve();
      }
      let pos = 0;
      // 每 0.1s 滚动 200px 直到到达底部，这两个数值来自若干实验，如微信文章中的图片
      const timer1 = setInterval(() => {
        pos += 200;
        if (pos > document.body.scrollHeight) {
          end();
        }
        window.scrollTo(0, pos);
      }, 100);
      // 20s 后无条件结束
      const timer2 = setTimeout(end, 20000);
    });
  });
}

async function main(browser, url) {
  const page = await browser.newPage();
  await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36');
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  try {
    await new Promise(r => setTimeout(r, 3000));
    // await Promise.race([
    //   new Promise(r => setTimeout(r, 5000)),
    //   page.waitForFunction(() => document.readyState === 'complete'),
    // ]);
  } catch {}
  await scroll(page);
  try {
    await new Promise(r => setTimeout(r, 3000));
    // await Promise.race([
    //   new Promise(r => setTimeout(r, 5000)),
    //   page.waitForFunction(() => document.readyState === 'complete'),
    // ]);
  } catch {}
  const cdp = await page.target().createCDPSession();
  const {root} = await cdp.send('DOM.getDocument');
  const {outerHTML: data} = await cdp.send('DOM.getOuterHTML', {nodeId: root.nodeId});
  let title = await page.title();
  return {title, data};
};

exports.handler = async (event, context) => {
  console.log({event, context});
  let browser = null;
  try {
    browser = await puppeteer.launch({
      args: chromium.args.concat([
        '--disable-file-system',
        '--window-size=1920,1080',
      ]),
      executablePath: await chromium.executablePath(),
      headless: chromium.headless,
      dumpio: true,
    });
    const {title, data} = await main(browser, event.url);
    return context.succeed({title, data});
  } catch (error) {
    return context.fail(error);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
};
