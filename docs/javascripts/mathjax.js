window.__mathjaxErrors = [];

window.MathJax = {
  startup: {
    typeset: false,
  },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[-]": ["noundefined"] },
    formatError: (jax, error) => {
      window.__mathjaxErrors.push(error.message);
      return jax.formatError(error);
    },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

let typesetPromise = Promise.resolve();

document$.subscribe(() => {
  if (!window.MathJax.typesetPromise) return;
  typesetPromise = typesetPromise
    .then(() => {
      MathJax.startup.output.clearCache();
      MathJax.typesetClear();
      MathJax.texReset();
      return MathJax.typesetPromise();
    })
    .catch((error) => {
      window.__mathjaxErrors.push(String(error));
      console.error(error);
    });
});
