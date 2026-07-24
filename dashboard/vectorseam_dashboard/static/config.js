/*
 * VectorSeam dashboard bootstrap.
 *
 * Loaded BEFORE support.js (the Claude Design runtime). It does two things:
 *
 *   1. Points the runtime at locally vendored React / ReactDOM / Babel instead
 *      of unpkg.com, via the runtime's `window.__resources` override. This is
 *      what makes the dashboard work offline / air-gapped and keeps a
 *      third-party CDN out of the runtime path. The vendored files are the
 *      exact upstream builds (their SHA-384 matches the runtime's pinned SRI).
 *
 *   2. Declares which calibration store and cohorts to render. The server
 *      rewrites the placeholders below from environment before serving this
 *      file, so a deployment configures the dashboard entirely through env
 *      (see server.py / VECTORSEAM_DASHBOARD_* variables). When cohorts is
 *      empty the component falls back to its bundled sample data.
 */
(function () {
  "use strict";

  var base = document.currentScript
    ? document.currentScript.src.replace(/\/config\.js(?:\?.*)?$/, "")
    : "";
  var vendor = base + "/vendor/";

  // URL-for-URL override consumed by the runtime's cdnScriptFor(). Keys must be
  // the exact unpkg URLs the runtime references; values are same-origin paths,
  // served without SRI (integrity is unnecessary and would block a local file).
  window.__resources = Object.assign({}, window.__resources, {
    "https://unpkg.com/react@18.3.1/umd/react.production.min.js":
      vendor + "react.production.min.js",
    "https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js":
      vendor + "react-dom.production.min.js",
    "https://unpkg.com/@babel/standalone@7.29.0/babel.min.js":
      vendor + "babel.min.js",
  });

  // Live wiring. The server also serves the calibration store under
  // /calibrations/<cohort>/… on this same origin. __DASHBOARD_CONFIG__ is
  // replaced by the server from environment; the fallback keeps the file usable
  // when opened directly (no server), in which case the component uses sample
  // data.
  var injected = "__DASHBOARD_CONFIG__";
  if (injected.charAt(0) === "{") {
    try {
      var cfg = JSON.parse(injected);
      // The component treats an empty baseUrl as "no live wiring" (it must be
      // truthy). Default it to this page's origin so same-origin deployments
      // fetch the store the server exposes here.
      if (!cfg.baseUrl) {
        cfg.baseUrl = window.location.origin;
      }
      window.VECTORSEAM_CONFIG = cfg;
    } catch (e) {
      /* leave unset -> sample data */
    }
  }
})();
