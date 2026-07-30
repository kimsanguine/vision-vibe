// 마법 지팡이 — opencv1_magic_wand/magic_wand.py 의 브라우저 이관.
//
// 원본은 OpenCV 고전 CV(HSV inRange → morphology → findContours → 무게중심)로
// 색상 물체를 추적하고, 페이드아웃 캔버스 + 무지개 궤적 + 글로우 + 헤일로 +
// 반짝이로 '마법' 연출을 얹는다. 여기서는 OpenCV 없이 Canvas 2D + 직접 픽셀
// 연산만으로 같은 결과를 만든다.
//
// ── 스코프 축소(원본과 의도적으로 다른 점) ─────────────────────────────
// 1) 중심점 계산: 원본은 cv2.findContours 로 '가장 큰 컨투어'의 무게중심을
//    쓰지만, 브라우저에는 동등한 함수가 없다. 이 포팅은 **임계값을 통과한
//    마스크 픽셀 전체의 평균 좌표**로 대체한다. 지배색 물체가 하나일 때는
//    사실상 동일하지만, 화면에 같은 색 물체가 여러 개 있으면 원본과 달리
//    두 물체의 중간점을 가리킬 수 있다.
// 2) 모폴로지(MORPH_OPEN/CLOSE) 생략: 점 노이즈 제거·구멍 메우기를 하지
//    않는다. 컨투어를 쓰지 않으므로 노이즈는 MIN_MASK_PIXELS 임계값과
//    평균화로만 완화된다 — 조명이 지저분하면 원본보다 중심점이 더 흔들린다.
// 3) 미러 모드 생략: 원본은 MIRROR=True(좌우 반전)로 지팡이 조작 직관성을
//    높이지만, 이 앱의 기존 탭들이 모두 반전 없이 그리므로 일관성을 택했다.
// 4) 글로우: cv2.GaussianBlur(sigma=9) + addWeighted 대신 Canvas 의
//    ctx.filter="blur(9px)" + 가산 합성('lighter')으로 근사한다.
//
// 노출 API(통합 단계에서 web/index.html 이 script 태그로 로드):
//   window.startMagicWand() / stopMagicWand() / clearMagicWandTrail() / cycleMagicWandColor()
// DOM 계약(통합 단계가 만들어 줌):
//   <video id="mp-video-magicwand">, <canvas id="mp-canvas-magicwand" width=640 height=480>
(function () {
  "use strict";

  // ── 설정 상수 (magic_wand.py 와 1:1) ───────────────────────────────
  const TRAIL_FADE = 0.93; // 원본 FADE — 매 프레임 궤적 감쇠율
  const TRAIL_THICKNESS = 5;
  const MAX_JUMP = 200; // 프레임 간 이동이 이보다 크면 선을 잇지 않음(오검출 방어)
  const GLOW_BLUR_PX = 9; // 원본 GLOW_SIGMA
  const GLOW_GAIN = 0.9; // 원본 GLOW_GAIN
  const HUE_STEP = 3; // 프레임당 무지개 진행량 (OpenCV H: 0~179)
  const HALO_RADIUS = 22;
  const SPARKLE_PER_FRAME = 2;
  const SPARKLE_LIFE = 12;
  const SAMPLE_STRIDE = 2; // 2픽셀 간격 샘플링 — 실시간 속도 확보
  // 원본 MIN_AREA=300(픽셀 면적). 2픽셀 격자로 샘플링하면 샘플 1개가 실제
  // 4픽셀을 대표하므로 300/4 = 75 를 임계값으로 쓴다.
  const MIN_MASK_PIXELS = 75;

  const VIDEO_ID = "mp-video-magicwand";
  const CANVAS_ID = "mp-canvas-magicwand";
  const STATUS_EVENT = "magicWandStatus";

  // OpenCV HSV 범위(H 0~179, S 0~255, V 0~255)를 그대로 쓴다.
  // 빨강은 색상환 0도 부근이라 구간이 끊겨 두 개의 hue 범위가 필요하다.
  const COLOR_PRESETS = [
    { name: "빨강", hueRanges: [[0, 10], [170, 179]], sat: [120, 255], val: [70, 255] },
    { name: "파랑", hueRanges: [[100, 130]], sat: [120, 255], val: [70, 255] },
    { name: "초록", hueRanges: [[40, 80]], sat: [90, 255], val: [70, 255] },
    { name: "노랑", hueRanges: [[22, 35]], sat: [120, 255], val: [100, 255] },
  ];

  // ── 1. 색상 추적 (RGB→HSV → 임계값 → 마스크 픽셀 평균) ─────────────
  /** RGB(0~255) → OpenCV 스케일 HSV [h:0~179, s:0~255, v:0~255]. */
  function rgbToHsv(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    if (d !== 0) {
      if (max === r) h = ((g - b) / d) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60;
      if (h < 0) h += 360;
    }
    const s = max === 0 ? 0 : d / max;
    // OpenCV 와 동일하게 각도를 2로 나눠 0~179 로 접는다(h*179/360 이 아니다 —
    // 파랑 240도가 119.33 이 아니라 정확히 120 이 되어야 프리셋 범위가 원본과 맞는다).
    return [h / 2, s * 255, max * 255];
  }

  function inHueRanges(h, ranges) {
    return ranges.some(([lo, hi]) => h >= lo && h <= hi);
  }

  /**
   * 프레임 ImageData 에서 프리셋 색상 마스크 픽셀들의 평균 좌표를 구한다.
   * findContours 없이 마스크 전체 평균으로 근사한다(파일 상단 스코프 축소 1번).
   * 통과 픽셀이 minPixels 미만이면 노이즈로 보고 null.
   */
  function findCentroid(imageData, preset, minPixels) {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    for (let y = 0; y < height; y += SAMPLE_STRIDE) {
      for (let x = 0; x < width; x += SAMPLE_STRIDE) {
        const i = (y * width + x) * 4;
        const hsv = rgbToHsv(data[i], data[i + 1], data[i + 2]);
        if (
          inHueRanges(hsv[0], preset.hueRanges) &&
          hsv[1] >= preset.sat[0] &&
          hsv[1] <= preset.sat[1] &&
          hsv[2] >= preset.val[0] &&
          hsv[2] <= preset.val[1]
        ) {
          sumX += x;
          sumY += y;
          count++;
        }
      }
    }
    if (count < minPixels) return null;
    return { x: sumX / count, y: sumY / count };
  }

  // ── 2. 마법 연출 (페이드 / 무지개 / 글로우 / 헤일로 / 반짝이) ────────
  /** 프레임 번호에 따라 순환하는 무지개 색상. OpenCV H(0~179) → CSS hsl(0~360). */
  function rainbowColor(tick) {
    const hue = (tick * HUE_STEP) % 180;
    return "hsl(" + hue * 2 + ", 100%, 60%)";
  }

  /**
   * 궤적 캔버스를 감쇠시킨다. 원본은 불투명 캔버스의 색상값을 0.93배 하지만,
   * Canvas 에서는 destination-in 으로 알파를 0.93배 하는 것이 등가다
   * (가산 합성 시 알파가 밝기로 작용).
   */
  function fadeTrail(trailCtx, width, height) {
    trailCtx.globalCompositeOperation = "destination-in";
    trailCtx.fillStyle = "rgba(0,0,0," + TRAIL_FADE + ")";
    trailCtx.fillRect(0, 0, width, height);
    trailCtx.globalCompositeOperation = "source-over";
  }

  /** 지팡이 끝을 감싸는 맥동하는 헤일로. 매 프레임 새로 그린다(누적 안 함). */
  function drawHalo(ctx, center, tick, color) {
    const pulse = HALO_RADIUS + 6 * Math.sin(tick * 0.25);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(center.x, center.y, pulse, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(center.x, center.y, Math.max(pulse - 8, 2), 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "#FFFFFF";
    ctx.beginPath();
    ctx.arc(center.x, center.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  /** 지팡이 끝에서 반짝이 파티클을 뿌린다. */
  function spawnSparkles(sparkles, center) {
    for (let i = 0; i < SPARKLE_PER_FRAME; i++) {
      sparkles.push({
        x: center.x,
        y: center.y,
        vx: (Math.random() - 0.5) * 5,
        vy: (Math.random() - 0.5) * 5,
        life: SPARKLE_LIFE,
      });
    }
  }

  /** 파티클을 이동시키며 그리고, 수명이 다한 것은 제거한다(배열 제자리 수정). */
  function updateSparkles(sparkles, ctx) {
    const alive = [];
    for (const s of sparkles) {
      s.x += s.vx;
      s.y += s.vy;
      s.life -= 1;
      if (s.life <= 0) continue;
      // 원본 BGR (shade, shade, 255) = 흰빛에서 붉은빛으로 사그라드는 색.
      const shade = Math.round((255 * s.life) / SPARKLE_LIFE);
      ctx.fillStyle = "rgb(255," + shade + "," + shade + ")";
      ctx.beginPath();
      ctx.arc(s.x, s.y, 2, 0, Math.PI * 2);
      ctx.fill();
      alive.push(s);
    }
    sparkles.length = 0;
    sparkles.push(...alive);
  }

  // ── 3. 컨트롤러 (카메라 루프 + 전역 함수 노출) ──────────────────────
  // 기존 web/index.html 의 createController 팩토리를 쓰지 않고 자체 루프를
  // 갖는다 — 이 파일이 index.html 의 module script 보다 먼저 로드될 수도 있어
  // window.createController 의존을 만들지 않기 위한 결정. 통합 단계에서 로딩
  // 순서가 확정되면 재사용으로 리팩터링할 수 있다.
  let presetIdx = 0;
  let trailCanvas = null;
  let trailCtx = null;
  let prevCenter = null;
  let sparkles = [];
  let hueTick = 0;
  let running = false;
  let rafId = null;
  let stream = null;

  function emitStatus(text) {
    window.dispatchEvent(new CustomEvent(STATUS_EVENT, { detail: { status: text } }));
  }

  function ensureTrailCanvas(width, height) {
    if (trailCanvas && trailCanvas.width === width && trailCanvas.height === height) return;
    trailCanvas = document.createElement("canvas");
    trailCanvas.width = width;
    trailCanvas.height = height;
    trailCtx = trailCanvas.getContext("2d");
  }

  /** 한 프레임을 추적·연출·합성하고, 상태 문구를 돌려준다. */
  function renderFrame(video, ctx, canvas) {
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.filter = "none";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const preset = COLOR_PRESETS[presetIdx];
    const center = findCentroid(frame, preset, MIN_MASK_PIXELS);

    // 궤적 레이어: 감쇠 → 새 선분 추가 (프레임 사이에 누적된다)
    ensureTrailCanvas(canvas.width, canvas.height);
    fadeTrail(trailCtx, canvas.width, canvas.height);
    if (center) {
      if (
        prevCenter &&
        Math.hypot(center.x - prevCenter.x, center.y - prevCenter.y) < MAX_JUMP
      ) {
        trailCtx.strokeStyle = rainbowColor(hueTick);
        trailCtx.lineWidth = TRAIL_THICKNESS;
        trailCtx.lineCap = "round";
        trailCtx.beginPath();
        trailCtx.moveTo(prevCenter.x, prevCenter.y);
        trailCtx.lineTo(center.x, center.y);
        trailCtx.stroke();
      }
      spawnSparkles(sparkles, center);
    }
    prevCenter = center;

    // 합성: 영상 + 글로우(블러 궤적) + 선명한 궤적 + 헤일로/반짝이.
    // 원본 cv2.add 와 같은 가산 합성을 쓴다.
    ctx.globalCompositeOperation = "lighter";
    if (typeof ctx.filter === "string") {
      // ctx.filter 미지원 브라우저에서는 글로우만 건너뛴다(궤적 자체는 보인다).
      ctx.filter = "blur(" + GLOW_BLUR_PX + "px)";
      ctx.globalAlpha = GLOW_GAIN;
      ctx.drawImage(trailCanvas, 0, 0);
      ctx.filter = "none";
      ctx.globalAlpha = 1;
    }
    ctx.drawImage(trailCanvas, 0, 0);

    // 헤일로·반짝이는 원본의 fx_layer 처럼 매 프레임 새로 그린다(누적 없음).
    if (center) drawHalo(ctx, center, hueTick, rainbowColor(hueTick));
    updateSparkles(sparkles, ctx);

    ctx.globalCompositeOperation = "source-over";
    hueTick++;

    ctx.font = "bold 16px sans-serif";
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText("지팡이 색상: " + preset.name, 12, 24);

    return center
      ? "지팡이 감지됨 (" + preset.name + ")"
      : "지팡이를 보여주세요 (" + preset.name + ")";
  }

  function releaseResources(video) {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    if (video) video.srcObject = null;
  }

  window.startMagicWand = async function () {
    if (running) return;
    const video = document.getElementById(VIDEO_ID);
    const canvas = document.getElementById(CANVAS_ID);
    const ctx = canvas.getContext("2d");
    running = true;
    try {
      emitStatus("카메라 권한 요청 중...");
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          "이 브라우저·주소에서는 카메라를 사용할 수 없습니다 (HTTPS 또는 localhost 필요)"
        );
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();

      const loop = () => {
        if (!running) return;
        try {
          if (video.readyState >= 2) {
            emitStatus(renderFrame(video, ctx, canvas));
          }
          rafId = requestAnimationFrame(loop);
        } catch (err) {
          running = false;
          releaseResources(video);
          emitStatus("오류: " + (err && err.message ? err.message : String(err)));
        }
      };
      rafId = requestAnimationFrame(loop);
    } catch (err) {
      running = false;
      releaseResources(video);
      emitStatus("오류: " + (err && err.message ? err.message : String(err)));
    }
  };

  window.stopMagicWand = function () {
    running = false;
    const video = document.getElementById(VIDEO_ID);
    const canvas = document.getElementById(CANVAS_ID);
    releaseResources(video);
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.filter = "none";
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    prevCenter = null;
    emitStatus("중지됨");
  };

  /** 원본 [s] 키 — 궤적 캔버스와 파티클을 함께 비운다. */
  window.clearMagicWandTrail = function () {
    if (trailCtx) trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
    sparkles = [];
    prevCenter = null;
  };

  /** 원본 [c] 키 — 추적 색상 프리셋을 순환한다. */
  window.cycleMagicWandColor = function () {
    presetIdx = (presetIdx + 1) % COLOR_PRESETS.length;
    prevCenter = null; // 색이 바뀌면 이전 좌표와 잇지 않는다
  };

  // 브라우저 콘솔 자체 검증용 노출 — 최종 사용자 기능이 아니다.
  window.__magicWandInternal = {
    rgbToHsv,
    inHueRanges,
    findCentroid,
    rainbowColor,
    COLOR_PRESETS,
    MIN_MASK_PIXELS,
  };
})();
