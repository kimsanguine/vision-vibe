// 문서 스캔 — opencv3_doc_scanner/scan.py 를 opencv.js(WASM)로 이관한 것.
//
// 파이프라인(원본과 동일):
//   축소 → 그레이스케일 → 가우시안 블러 → auto-Canny → dilate
//   → findContours → 면적 최대 4각형(approxPolyDP) → order_points
//   → getPerspectiveTransform → warpPerspective
//
// 스코프 축소(명시): 원본 scan.py 의 estimate_aspect_ratio(소실점 기반 종횡비
// 복원)는 이관하지 않는다. four_point_transform 의 기본 휴리스틱 — 마주보는 두
// 변 중 긴 쪽을 출력 크기로 쓰는 방식 — 만 포팅한다. 따라서 비스듬한 각도에서
// 찍은 문서는 원본 파이썬 버전보다 종횡비가 덜 정확할 수 있다.
//
// 원본과 다르게 처리한 지점 2곳(둘 다 opencv.js 의 API 제약 때문):
//   1) 밝기 중앙값: cv.calcHist 대신 Mat.data 를 직접 256-bin 히스토그램으로
//      센다. opencv.js 의 MatVector 는 배열 리터럴로 생성할 수 없어
//      calcHist 호출부가 장황해지는데, 중앙값 계산은 결과가 동일하다.
//   2) dilate: 결과를 별도 Mat 에 받는다. opencv.js 에서 src===dst 인 in-place
//      호출은 보장되지 않는다.
(function () {
  "use strict";

  // 엣지 검출은 축소 이미지에서 수행한다 (속도 + 잡은 노이즈 억제).
  // 찾은 꼭짓점은 원본 좌표로 되돌려 원본 해상도로 보정한다.
  const DETECT_WIDTH = 600;

  // 문서 후보 4각형이 전체 이미지에서 차지해야 하는 최소 면적 비율.
  // 이보다 작으면 문서가 아니라 이미지 안의 작은 사각형(로고·표 등)으로 본다.
  const MIN_AREA_RATIO = 0.15;

  // scan.py 와 동일하게 면적 상위 5개 컨투어만 4각형 후보로 검사한다.
  const MAX_CONTOUR_CANDIDATES = 5;

  const STATUS_EVENT = "docScannerStatus";
  const RESULT_EVENT = "docScanResult";

  const VIDEO_ID = "mp-video-docscan";
  const LIVE_CANVAS_ID = "mp-canvas-docscan";
  const RESULT_CANVAS_ID = "mp-canvas-docscan-result";

  function emitStatus(text) {
    window.dispatchEvent(
      new CustomEvent(STATUS_EVENT, { detail: { status: text } })
    );
  }

  /// opencv.js 는 async 로드라 사용자가 버튼을 먼저 누를 수 있다.
  /// 로드 전이면 조용히 실패하지 않고 상태 문구로 알린다.
  function isCvReady() {
    return typeof cv !== "undefined" && typeof cv.Mat === "function";
  }

  /// 단일 채널 8비트 Mat 의 밝기 중앙값. scan.py 의 np.median(blurred) 대응.
  function medianOfMat(mat) {
    const data = mat.data;
    const histogram = new Uint32Array(256);
    for (let i = 0; i < data.length; i++) {
      histogram[data[i]]++;
    }
    const half = data.length / 2;
    let cumulative = 0;
    for (let value = 0; value < 256; value++) {
      cumulative += histogram[value];
      if (cumulative >= half) return value;
    }
    return 128;
  }

  /// scan.py 의 order_points 이관 — 좌상단→우상단→우하단→좌하단 순 정렬.
  ///
  /// findContours 가 주는 점 순서는 보장되지 않으므로, 정렬 없이
  /// getPerspectiveTransform 에 넣으면 결과가 뒤집히거나 대각으로 꼬인다.
  ///
  /// 판별 근거: x+y 최소 = 좌상단, 최대 = 우하단 / y-x 최소 = 우상단, 최대 = 좌하단
  ///
  /// 같은 점이 두 번 배정되면 정렬 실패다 — 원본이 예외를 던지는 자리이므로
  /// 여기서도 조용히 넘기지 않고 null 을 돌려 호출측이 실패로 보고하게 한다.
  function orderPoints(points) {
    const sums = points.map(([x, y]) => x + y);
    const diffs = points.map(([x, y]) => y - x);

    const tlIndex = sums.indexOf(Math.min.apply(null, sums));
    const brIndex = sums.indexOf(Math.max.apply(null, sums));
    const trIndex = diffs.indexOf(Math.min.apply(null, diffs));
    const blIndex = diffs.indexOf(Math.max.apply(null, diffs));

    const chosen = [tlIndex, trIndex, brIndex, blIndex];
    if (new Set(chosen).size !== 4) return null;

    return chosen.map((index) => points[index]);
  }

  /// RGBA cv.Mat → 원본 좌표계의 정렬된 4점 [[x,y] x4], 못 찾으면 null.
  /// scan.py 의 detect_edges + find_document_quad + 좌표 복원을 합친 것.
  function findDocumentQuad(srcMat) {
    const scale = DETECT_WIDTH / srcMat.cols;
    const smallHeight = Math.max(Math.round(srcMat.rows * scale), 1);

    const small = new cv.Mat();
    const gray = new cv.Mat();
    const blurred = new cv.Mat();
    const edges = new cv.Mat();
    const dilated = new cv.Mat();
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();

    try {
      cv.resize(srcMat, small, new cv.Size(DETECT_WIDTH, smallHeight));
      cv.cvtColor(small, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

      // 조명·대비가 이미지마다 달라 고정 임계값은 잘 깨진다.
      // 밝기 중앙값 기준 자동 임계값(auto-Canny)으로 잡는다.
      const median = medianOfMat(blurred);
      const lower = Math.max(0, Math.round(0.66 * median));
      const upper = Math.min(255, Math.round(1.33 * median));
      cv.Canny(blurred, edges, lower, upper);

      // 종이 경계가 조명 때문에 끊기면 contour 가 닫히지 않아 4각형을 못 찾는다.
      // 살짝 팽창시켜 틈을 메운다.
      cv.dilate(edges, dilated, kernel);

      cv.findContours(
        dilated,
        contours,
        hierarchy,
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE
      );

      // scan.py 와 동일하게 "면적 상위 5개 중 조건을 만족하는 첫 번째"를 고른다.
      // MatVector.get() 은 매번 새 래퍼를 돌려주므로 쓰고 나면 반드시 delete 한다.
      const ranked = [];
      for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        ranked.push({ index: i, area: cv.contourArea(contour) });
        contour.delete();
      }
      ranked.sort((a, b) => b.area - a.area);

      const imageArea = small.rows * small.cols;
      let quad = null;

      for (const candidate of ranked.slice(0, MAX_CONTOUR_CANDIDATES)) {
        const contour = contours.get(candidate.index);
        const approx = new cv.Mat();
        try {
          const perimeter = cv.arcLength(contour, true);
          // epsilon = 둘레의 2% — 종이의 미세한 굴곡은 무시하고 직선으로 근사
          cv.approxPolyDP(contour, approx, 0.02 * perimeter, true);

          if (approx.rows !== 4) continue;
          if (cv.contourArea(approx) < MIN_AREA_RATIO * imageArea) continue;
          if (!cv.isContourConvex(approx)) continue;

          // 축소 좌표에서 찾은 꼭짓점을 원본 해상도로 되돌린다.
          const points = [];
          for (let i = 0; i < 4; i++) {
            points.push([
              approx.data32S[i * 2] / scale,
              approx.data32S[i * 2 + 1] / scale,
            ]);
          }
          quad = orderPoints(points);
          if (quad) break;
        } finally {
          approx.delete();
          contour.delete();
        }
      }

      return quad;
    } finally {
      small.delete();
      gray.delete();
      blurred.delete();
      edges.delete();
      dilated.delete();
      kernel.delete();
      contours.delete();
      hierarchy.delete();
    }
  }

  /// scan.py 의 four_point_transform 이관.
  /// 소실점 종횡비 보정(estimate_aspect_ratio)은 제외 — 파일 상단 스코프 축소 참고.
  /// 반환한 cv.Mat 의 delete() 책임은 호출측에 있다.
  function fourPointTransform(srcMat, quad) {
    const [tl, tr, br, bl] = quad;
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

    // 마주보는 두 변 중 긴 쪽을 출력 크기로 잡는다
    // (원근으로 짧아진 쪽 기준이면 찌그러진다).
    const width = Math.max(dist(br, bl), dist(tr, tl));
    const height = Math.max(dist(tr, br), dist(tl, bl));
    const outW = Math.max(Math.round(width), 1);
    const outH = Math.max(Math.round(height), 1);

    const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
      tl[0], tl[1],
      tr[0], tr[1],
      br[0], br[1],
      bl[0], bl[1],
    ]);
    const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      outW - 1, 0,
      outW - 1, outH - 1,
      0, outH - 1,
    ]);
    const matrix = cv.getPerspectiveTransform(srcTri, dstTri);
    const warped = new cv.Mat();

    try {
      cv.warpPerspective(srcMat, warped, matrix, new cv.Size(outW, outH));
    } finally {
      srcTri.delete();
      dstTri.delete();
      matrix.delete();
    }
    return warped;
  }

  let stream = null;

  window.startDocScanner = async function () {
    const video = document.getElementById(VIDEO_ID);
    if (!video) {
      emitStatus("오류: 카메라 영역(#" + VIDEO_ID + ")을 찾지 못했습니다.");
      return;
    }
    if (stream) return;
    try {
      emitStatus("카메라 권한 요청 중...");
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      emitStatus("문서를 화면에 맞추고 스캔 버튼을 누르세요");
    } catch (err) {
      stream = null;
      emitStatus("오류: " + (err && err.message ? err.message : String(err)));
    }
  };

  window.stopDocScanner = function () {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    const video = document.getElementById(VIDEO_ID);
    if (video) video.srcObject = null;
    emitStatus("중지됨");
  };

  window.captureAndScan = function () {
    const video = document.getElementById(VIDEO_ID);
    const liveCanvas = document.getElementById(LIVE_CANVAS_ID);
    const resultCanvas = document.getElementById(RESULT_CANVAS_ID);
    if (!video || !liveCanvas || !resultCanvas) {
      emitStatus("오류: 스캔 화면 요소를 찾지 못했습니다.");
      return;
    }
    if (!isCvReady()) {
      emitStatus("이미지 처리 엔진을 아직 불러오는 중입니다. 잠시 후 다시 눌러주세요.");
      return;
    }
    if (video.readyState < 2) {
      emitStatus("카메라가 아직 준비되지 않았습니다. 먼저 시작을 눌러주세요.");
      return;
    }

    liveCanvas
      .getContext("2d")
      .drawImage(video, 0, 0, liveCanvas.width, liveCanvas.height);

    let src = null;
    let warped = null;
    try {
      src = cv.imread(liveCanvas);
      const quad = findDocumentQuad(src);
      if (!quad) {
        emitStatus(
          "문서 4각형을 찾지 못했습니다. 배경과 대비가 잘 되는 곳에서 다시 시도하세요."
        );
        return;
      }

      warped = fourPointTransform(src, quad);
      resultCanvas.width = warped.cols;
      resultCanvas.height = warped.rows;
      cv.imshow(resultCanvas, warped);

      emitStatus(
        "스캔 완료 (" + resultCanvas.width + "x" + resultCanvas.height + "px)"
      );
      window.dispatchEvent(
        new CustomEvent(RESULT_EVENT, {
          detail: {
            scanned: true,
            width: resultCanvas.width,
            height: resultCanvas.height,
          },
        })
      );
    } catch (err) {
      emitStatus(
        "스캔 실패: " + (err && err.message ? err.message : String(err))
      );
    } finally {
      if (src) src.delete();
      if (warped) warped.delete();
    }
  };

  // 브라우저 콘솔에서 개별 단계를 직접 검증하기 위한 노출 — 최종 사용자 기능이 아니다.
  window.__docScannerInternal = {
    findDocumentQuad,
    orderPoints,
    medianOfMat,
    fourPointTransform,
    isCvReady,
  };
})();
