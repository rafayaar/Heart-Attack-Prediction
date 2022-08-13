# Heart-Attack-Prediction
Training and visualizing data of heart attack patients 



<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Heart Attack Prediction</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>




<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>



<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/*
 * Webkit scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-corner {
  background: var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-thumb {
  background: rgb(var(--jp-scrollbar-thumb-color));
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-right: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-bottom: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar */

[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-corner,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-corner {
  background-color: transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-thumb,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid transparent;
  border-right: var(--jp-scrollbar-endpad) solid transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid transparent;
  border-bottom: var(--jp-scrollbar-endpad) solid transparent;
}

/*
 * Phosphor
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Widget, /* </DEPRECATED> */
.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  cursor: default;
}


/* <DEPRECATED> */ .p-Widget.p-mod-hidden, /* </DEPRECATED> */
.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-CommandPalette, /* </DEPRECATED> */
.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-CommandPalette-search, /* </DEPRECATED> */
.lm-CommandPalette-search {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-content, /* </DEPRECATED> */
.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-CommandPalette-header, /* </DEPRECATED> */
.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}


/* <DEPRECATED> */ .p-CommandPalette-item, /* </DEPRECATED> */
.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}


/* <DEPRECATED> */ .p-CommandPalette-itemIcon, /* </DEPRECATED> */
.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemContent, /* </DEPRECATED> */
.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}


/* <DEPRECATED> */ .p-CommandPalette-itemShortcut, /* </DEPRECATED> */
.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemLabel, /* </DEPRECATED> */
.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-DockPanel, /* </DEPRECATED> */
.lm-DockPanel {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-widget, /* </DEPRECATED> */
.lm-DockPanel-widget {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-tabBar, /* </DEPRECATED> */
.lm-DockPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-DockPanel-handle, /* </DEPRECATED> */
.lm-DockPanel-handle {
  z-index: 2;
}


/* <DEPRECATED> */ .p-DockPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-DockPanel-handle:after, /* </DEPRECATED> */
.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}


/* <DEPRECATED> */ .p-DockPanel-overlay, /* </DEPRECATED> */
.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}


/* <DEPRECATED> */ .p-DockPanel-overlay.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Menu, /* </DEPRECATED> */
.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-Menu-content, /* </DEPRECATED> */
.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-Menu-item, /* </DEPRECATED> */
.lm-Menu-item {
  display: table-row;
}


/* <DEPRECATED> */
.p-Menu-item.p-mod-hidden,
.p-Menu-item.p-mod-collapsed,
/* </DEPRECATED> */
.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}


/* <DEPRECATED> */
.p-Menu-itemIcon,
.p-Menu-itemSubmenuIcon,
/* </DEPRECATED> */
.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}


/* <DEPRECATED> */ .p-Menu-itemLabel, /* </DEPRECATED> */
.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}


/* <DEPRECATED> */ .p-Menu-itemShortcut, /* </DEPRECATED> */
.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-MenuBar, /* </DEPRECATED> */
.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-MenuBar-content, /* </DEPRECATED> */
.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}


/* <DEPRECATED> */ .p--MenuBar-item, /* </DEPRECATED> */
.lm-MenuBar-item {
  box-sizing: border-box;
}


/* <DEPRECATED> */
.p-MenuBar-itemIcon,
.p-MenuBar-itemLabel,
/* </DEPRECATED> */
.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-ScrollBar, /* </DEPRECATED> */
.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-ScrollBar-button, /* </DEPRECATED> */
.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-track, /* </DEPRECATED> */
.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-thumb, /* </DEPRECATED> */
.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-SplitPanel-child, /* </DEPRECATED> */
.lm-SplitPanel-child {
  z-index: 0;
}


/* <DEPRECATED> */ .p-SplitPanel-handle, /* </DEPRECATED> */
.lm-SplitPanel-handle {
  z-index: 1;
}


/* <DEPRECATED> */ .p-SplitPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-SplitPanel-handle:after, /* </DEPRECATED> */
.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabBar, /* </DEPRECATED> */
.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-content, /* </DEPRECATED> */
.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='horizontal'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='vertical'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}


/* <DEPRECATED> */
.p-TabBar-tabIcon,
.p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-TabBar-tabLabel, /* </DEPRECATED> */
.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}


/* <DEPRECATED> */ .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabPanel-tabBar, /* </DEPRECATED> */
.lm-TabPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-TabPanel-stackedPanel, /* </DEPRECATED> */
.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

@charset "UTF-8";
/*!

Copyright 2015-present Palantir Technologies, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.

*/
html{
  -webkit-box-sizing:border-box;
          box-sizing:border-box; }

*,
*::before,
*::after{
  -webkit-box-sizing:inherit;
          box-sizing:inherit; }

body{
  text-transform:none;
  line-height:1.28581;
  letter-spacing:0;
  font-size:14px;
  font-weight:400;
  color:#182026;
  font-family:-apple-system, "BlinkMacSystemFont", "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Open Sans", "Helvetica Neue", "Icons16", sans-serif; }

p{
  margin-top:0;
  margin-bottom:10px; }

small{
  font-size:12px; }

strong{
  font-weight:600; }

::-moz-selection{
  background:rgba(125, 188, 255, 0.6); }

::selection{
  background:rgba(125, 188, 255, 0.6); }
.bp3-heading{
  color:#182026;
  font-weight:600;
  margin:0 0 10px;
  padding:0; }
  .bp3-dark .bp3-heading{
    color:#f5f8fa; }

h1.bp3-heading, .bp3-running-text h1{
  line-height:40px;
  font-size:36px; }

h2.bp3-heading, .bp3-running-text h2{
  line-height:32px;
  font-size:28px; }

h3.bp3-heading, .bp3-running-text h3{
  line-height:25px;
  font-size:22px; }

h4.bp3-heading, .bp3-running-text h4{
  line-height:21px;
  font-size:18px; }

h5.bp3-heading, .bp3-running-text h5{
  line-height:19px;
  font-size:16px; }

h6.bp3-heading, .bp3-running-text h6{
  line-height:16px;
  font-size:14px; }
.bp3-ui-text{
  text-transform:none;
  line-height:1.28581;
  letter-spacing:0;
  font-size:14px;
  font-weight:400; }

.bp3-monospace-text{
  text-transform:none;
  font-family:monospace; }

.bp3-text-muted{
  color:#5c7080; }
  .bp3-dark .bp3-text-muted{
    color:#a7b6c2; }

.bp3-text-disabled{
  color:rgba(92, 112, 128, 0.6); }
  .bp3-dark .bp3-text-disabled{
    color:rgba(167, 182, 194, 0.6); }

.bp3-text-overflow-ellipsis{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal; }
.bp3-running-text{
  line-height:1.5;
  font-size:14px; }
  .bp3-running-text h1{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h1{
      color:#f5f8fa; }
  .bp3-running-text h2{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h2{
      color:#f5f8fa; }
  .bp3-running-text h3{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h3{
      color:#f5f8fa; }
  .bp3-running-text h4{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h4{
      color:#f5f8fa; }
  .bp3-running-text h5{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h5{
      color:#f5f8fa; }
  .bp3-running-text h6{
    color:#182026;
    font-weight:600;
    margin-top:40px;
    margin-bottom:20px; }
    .bp3-dark .bp3-running-text h6{
      color:#f5f8fa; }
  .bp3-running-text hr{
    margin:20px 0;
    border:none;
    border-bottom:1px solid rgba(16, 22, 26, 0.15); }
    .bp3-dark .bp3-running-text hr{
      border-color:rgba(255, 255, 255, 0.15); }
  .bp3-running-text p{
    margin:0 0 10px;
    padding:0; }

.bp3-text-large{
  font-size:16px; }

.bp3-text-small{
  font-size:12px; }
a{
  text-decoration:none;
  color:#106ba3; }
  a:hover{
    cursor:pointer;
    text-decoration:underline;
    color:#106ba3; }
  a .bp3-icon, a .bp3-icon-standard, a .bp3-icon-large{
    color:inherit; }
  a code,
  .bp3-dark a code{
    color:inherit; }
  .bp3-dark a,
  .bp3-dark a:hover{
    color:#48aff0; }
    .bp3-dark a .bp3-icon, .bp3-dark a .bp3-icon-standard, .bp3-dark a .bp3-icon-large,
    .bp3-dark a:hover .bp3-icon,
    .bp3-dark a:hover .bp3-icon-standard,
    .bp3-dark a:hover .bp3-icon-large{
      color:inherit; }
.bp3-running-text code, .bp3-code{
  text-transform:none;
  font-family:monospace;
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
  background:rgba(255, 255, 255, 0.7);
  padding:2px 5px;
  color:#5c7080;
  font-size:smaller; }
  .bp3-dark .bp3-running-text code, .bp3-running-text .bp3-dark code, .bp3-dark .bp3-code{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#a7b6c2; }
  .bp3-running-text a > code, a > .bp3-code{
    color:#137cbd; }
    .bp3-dark .bp3-running-text a > code, .bp3-running-text .bp3-dark a > code, .bp3-dark a > .bp3-code{
      color:inherit; }

.bp3-running-text pre, .bp3-code-block{
  text-transform:none;
  font-family:monospace;
  display:block;
  margin:10px 0;
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
  background:rgba(255, 255, 255, 0.7);
  padding:13px 15px 12px;
  line-height:1.4;
  color:#182026;
  font-size:13px;
  word-break:break-all;
  word-wrap:break-word; }
  .bp3-dark .bp3-running-text pre, .bp3-running-text .bp3-dark pre, .bp3-dark .bp3-code-block{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#f5f8fa; }
  .bp3-running-text pre > code, .bp3-code-block > code{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:none;
    padding:0;
    color:inherit;
    font-size:inherit; }

.bp3-running-text kbd, .bp3-key{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  background:#ffffff;
  min-width:24px;
  height:24px;
  padding:3px 6px;
  vertical-align:middle;
  line-height:24px;
  color:#5c7080;
  font-family:inherit;
  font-size:12px; }
  .bp3-running-text kbd .bp3-icon, .bp3-key .bp3-icon, .bp3-running-text kbd .bp3-icon-standard, .bp3-key .bp3-icon-standard, .bp3-running-text kbd .bp3-icon-large, .bp3-key .bp3-icon-large{
    margin-right:5px; }
  .bp3-dark .bp3-running-text kbd, .bp3-running-text .bp3-dark kbd, .bp3-dark .bp3-key{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
    background:#394b59;
    color:#a7b6c2; }
.bp3-running-text blockquote, .bp3-blockquote{
  margin:0 0 10px;
  border-left:solid 4px rgba(167, 182, 194, 0.5);
  padding:0 20px; }
  .bp3-dark .bp3-running-text blockquote, .bp3-running-text .bp3-dark blockquote, .bp3-dark .bp3-blockquote{
    border-color:rgba(115, 134, 148, 0.5); }
.bp3-running-text ul,
.bp3-running-text ol, .bp3-list{
  margin:10px 0;
  padding-left:30px; }
  .bp3-running-text ul li:not(:last-child), .bp3-running-text ol li:not(:last-child), .bp3-list li:not(:last-child){
    margin-bottom:5px; }
  .bp3-running-text ul ol, .bp3-running-text ol ol, .bp3-list ol,
  .bp3-running-text ul ul,
  .bp3-running-text ol ul,
  .bp3-list ul{
    margin-top:5px; }

.bp3-list-unstyled{
  margin:0;
  padding:0;
  list-style:none; }
  .bp3-list-unstyled li{
    padding:0; }
.bp3-rtl{
  text-align:right; }

.bp3-dark{
  color:#f5f8fa; }

:focus{
  outline:rgba(19, 124, 189, 0.6) auto 2px;
  outline-offset:2px;
  -moz-outline-radius:6px; }

.bp3-focus-disabled :focus{
  outline:none !important; }
  .bp3-focus-disabled :focus ~ .bp3-control-indicator{
    outline:none !important; }

.bp3-alert{
  max-width:400px;
  padding:20px; }

.bp3-alert-body{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-alert-body .bp3-icon{
    margin-top:0;
    margin-right:20px;
    font-size:40px; }

.bp3-alert-footer{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:reverse;
      -ms-flex-direction:row-reverse;
          flex-direction:row-reverse;
  margin-top:10px; }
  .bp3-alert-footer .bp3-button{
    margin-left:10px; }
.bp3-breadcrumbs{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:wrap;
      flex-wrap:wrap;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  margin:0;
  cursor:default;
  height:30px;
  padding:0;
  list-style:none; }
  .bp3-breadcrumbs > li{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center; }
    .bp3-breadcrumbs > li::after{
      display:block;
      margin:0 5px;
      background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M10.71 7.29l-4-4a1.003 1.003 0 0 0-1.42 1.42L8.59 8 5.3 11.29c-.19.18-.3.43-.3.71a1.003 1.003 0 0 0 1.71.71l4-4c.18-.18.29-.43.29-.71 0-.28-.11-.53-.29-.71z' fill='%235C7080'/%3e%3c/svg%3e");
      width:16px;
      height:16px;
      content:""; }
    .bp3-breadcrumbs > li:last-of-type::after{
      display:none; }

.bp3-breadcrumb,
.bp3-breadcrumb-current,
.bp3-breadcrumbs-collapsed{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  font-size:16px; }

.bp3-breadcrumb,
.bp3-breadcrumbs-collapsed{
  color:#5c7080; }

.bp3-breadcrumb:hover{
  text-decoration:none; }

.bp3-breadcrumb.bp3-disabled{
  cursor:not-allowed;
  color:rgba(92, 112, 128, 0.6); }

.bp3-breadcrumb .bp3-icon{
  margin-right:5px; }

.bp3-breadcrumb-current{
  color:inherit;
  font-weight:600; }
  .bp3-breadcrumb-current .bp3-input{
    vertical-align:baseline;
    font-size:inherit;
    font-weight:inherit; }

.bp3-breadcrumbs-collapsed{
  margin-right:2px;
  border:none;
  border-radius:3px;
  background:#ced9e0;
  cursor:pointer;
  padding:1px 5px;
  vertical-align:text-bottom; }
  .bp3-breadcrumbs-collapsed::before{
    display:block;
    background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cg fill='%235C7080'%3e%3ccircle cx='2' cy='8.03' r='2'/%3e%3ccircle cx='14' cy='8.03' r='2'/%3e%3ccircle cx='8' cy='8.03' r='2'/%3e%3c/g%3e%3c/svg%3e") center no-repeat;
    width:16px;
    height:16px;
    content:""; }
  .bp3-breadcrumbs-collapsed:hover{
    background:#bfccd6;
    text-decoration:none;
    color:#182026; }

.bp3-dark .bp3-breadcrumb,
.bp3-dark .bp3-breadcrumbs-collapsed{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumbs > li::after{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumb.bp3-disabled{
  color:rgba(167, 182, 194, 0.6); }

.bp3-dark .bp3-breadcrumb-current{
  color:#f5f8fa; }

.bp3-dark .bp3-breadcrumbs-collapsed{
  background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-breadcrumbs-collapsed:hover{
    background:rgba(16, 22, 26, 0.6);
    color:#f5f8fa; }
.bp3-button{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  padding:5px 10px;
  vertical-align:middle;
  text-align:left;
  font-size:14px;
  min-width:30px;
  min-height:30px; }
  .bp3-button > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-button > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-button::before,
  .bp3-button > *{
    margin-right:7px; }
  .bp3-button:empty::before,
  .bp3-button > :last-child{
    margin-right:0; }
  .bp3-button:empty{
    padding:0 !important; }
  .bp3-button:disabled, .bp3-button.bp3-disabled{
    cursor:not-allowed; }
  .bp3-button.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button.bp3-align-right,
  .bp3-align-right .bp3-button{
    text-align:right; }
  .bp3-button.bp3-align-left,
  .bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-button:not([class*="bp3-intent-"]){
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    color:#182026; }
    .bp3-button:not([class*="bp3-intent-"]):hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
      background-clip:padding-box;
      background-color:#ebf1f5; }
    .bp3-button:not([class*="bp3-intent-"]):active, .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#d8e1e8;
      background-image:none; }
    .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      outline:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6); }
      .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active:hover, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-button.bp3-intent-primary{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover, .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      background-color:#106ba3; }
    .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#0e5a8a;
      background-image:none; }
    .bp3-button.bp3-intent-primary:disabled, .bp3-button.bp3-intent-primary.bp3-disabled{
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(19, 124, 189, 0.5);
      background-image:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-success{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#0f9960;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
    .bp3-button.bp3-intent-success:hover, .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-success:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      background-color:#0d8050; }
    .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#0a6640;
      background-image:none; }
    .bp3-button.bp3-intent-success:disabled, .bp3-button.bp3-intent-success.bp3-disabled{
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(15, 153, 96, 0.5);
      background-image:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-warning{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#d9822b;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover, .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      background-color:#bf7326; }
    .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#a66321;
      background-image:none; }
    .bp3-button.bp3-intent-warning:disabled, .bp3-button.bp3-intent-warning.bp3-disabled{
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(217, 130, 43, 0.5);
      background-image:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-danger{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#db3737;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover, .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      background-color:#c23030; }
    .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#a82a2a;
      background-image:none; }
    .bp3-button.bp3-intent-danger:disabled, .bp3-button.bp3-intent-danger.bp3-disabled{
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(219, 55, 55, 0.5);
      background-image:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
    stroke:#ffffff; }
  .bp3-button.bp3-large,
  .bp3-large .bp3-button{
    min-width:40px;
    min-height:40px;
    padding:5px 15px;
    font-size:16px; }
    .bp3-button.bp3-large::before,
    .bp3-button.bp3-large > *,
    .bp3-large .bp3-button::before,
    .bp3-large .bp3-button > *{
      margin-right:10px; }
    .bp3-button.bp3-large:empty::before,
    .bp3-button.bp3-large > :last-child,
    .bp3-large .bp3-button:empty::before,
    .bp3-large .bp3-button > :last-child{
      margin-right:0; }
  .bp3-button.bp3-small,
  .bp3-small .bp3-button{
    min-width:24px;
    min-height:24px;
    padding:0 7px; }
  .bp3-button.bp3-loading{
    position:relative; }
    .bp3-button.bp3-loading[class*="bp3-icon-"]::before{
      visibility:hidden; }
    .bp3-button.bp3-loading .bp3-button-spinner{
      position:absolute;
      margin:0; }
    .bp3-button.bp3-loading > :not(.bp3-button-spinner){
      visibility:hidden; }
  .bp3-button[class*="bp3-icon-"]::before{
    line-height:1;
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-weight:400;
    font-style:normal;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    color:#5c7080; }
  .bp3-button .bp3-icon, .bp3-button .bp3-icon-standard, .bp3-button .bp3-icon-large{
    color:#5c7080; }
    .bp3-button .bp3-icon.bp3-align-right, .bp3-button .bp3-icon-standard.bp3-align-right, .bp3-button .bp3-icon-large.bp3-align-right{
      margin-left:7px; }
  .bp3-button .bp3-icon:first-child:last-child,
  .bp3-button .bp3-spinner + .bp3-icon:last-child{
    margin:0 -7px; }
  .bp3-dark .bp3-button:not([class*="bp3-intent-"]){
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover, .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#30404d; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#202b33;
      background-image:none; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"])[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-large{
      color:#a7b6c2; }
  .bp3-dark .bp3-button[class*="bp3-intent-"]{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:active, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:disabled, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background-image:none;
      color:rgba(255, 255, 255, 0.3); }
    .bp3-dark .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
      stroke:#8a9ba8; }
  .bp3-button:disabled::before,
  .bp3-button:disabled .bp3-icon, .bp3-button:disabled .bp3-icon-standard, .bp3-button:disabled .bp3-icon-large, .bp3-button.bp3-disabled::before,
  .bp3-button.bp3-disabled .bp3-icon, .bp3-button.bp3-disabled .bp3-icon-standard, .bp3-button.bp3-disabled .bp3-icon-large, .bp3-button[class*="bp3-intent-"]::before,
  .bp3-button[class*="bp3-intent-"] .bp3-icon, .bp3-button[class*="bp3-intent-"] .bp3-icon-standard, .bp3-button[class*="bp3-intent-"] .bp3-icon-large{
    color:inherit !important; }
  .bp3-button.bp3-minimal{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:none; }
    .bp3-button.bp3-minimal:hover{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(167, 182, 194, 0.3);
      text-decoration:none;
      color:#182026; }
    .bp3-button.bp3-minimal:active, .bp3-button.bp3-minimal.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(115, 134, 148, 0.3);
      color:#182026; }
    .bp3-button.bp3-minimal:disabled, .bp3-button.bp3-minimal:disabled:hover, .bp3-button.bp3-minimal.bp3-disabled, .bp3-button.bp3-minimal.bp3-disabled:hover{
      background:none;
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6); }
      .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-minimal{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-minimal:hover, .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none; }
      .bp3-dark .bp3-button.bp3-minimal:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-minimal:disabled, .bp3-dark .bp3-button.bp3-minimal:disabled:hover, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover{
        background:none;
        cursor:not-allowed;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover, .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover, .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover, .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover, .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }

a.bp3-button{
  text-align:center;
  text-decoration:none;
  -webkit-transition:none;
  transition:none; }
  a.bp3-button, a.bp3-button:hover, a.bp3-button:active{
    color:#182026; }
  a.bp3-button.bp3-disabled{
    color:rgba(92, 112, 128, 0.6); }

.bp3-button-text{
  -webkit-box-flex:0;
      -ms-flex:0 1 auto;
          flex:0 1 auto; }

.bp3-button.bp3-align-left .bp3-button-text, .bp3-button.bp3-align-right .bp3-button-text,
.bp3-button-group.bp3-align-left .bp3-button-text,
.bp3-button-group.bp3-align-right .bp3-button-text{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto; }
.bp3-button-group{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex; }
  .bp3-button-group .bp3-button{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    position:relative;
    z-index:4; }
    .bp3-button-group .bp3-button:focus{
      z-index:5; }
    .bp3-button-group .bp3-button:hover{
      z-index:6; }
    .bp3-button-group .bp3-button:active, .bp3-button-group .bp3-button.bp3-active{
      z-index:7; }
    .bp3-button-group .bp3-button:disabled, .bp3-button-group .bp3-button.bp3-disabled{
      z-index:3; }
    .bp3-button-group .bp3-button[class*="bp3-intent-"]{
      z-index:9; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:focus{
        z-index:10; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:hover{
        z-index:11; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:active, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-active{
        z-index:12; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:disabled, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-disabled{
        z-index:8; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:first-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:first-child){
    border-top-left-radius:0;
    border-bottom-left-radius:0; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    margin-right:-1px;
    border-top-right-radius:0;
    border-bottom-right-radius:0; }
  .bp3-button-group.bp3-minimal .bp3-button{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:none; }
    .bp3-button-group.bp3-minimal .bp3-button:hover{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(167, 182, 194, 0.3);
      text-decoration:none;
      color:#182026; }
    .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(115, 134, 148, 0.3);
      color:#182026; }
    .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
      background:none;
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6); }
      .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button-group.bp3-minimal .bp3-button{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:inherit; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
        background:none;
        cursor:not-allowed;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
      color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
      color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:none;
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button-group .bp3-popover-wrapper,
  .bp3-button-group .bp3-popover-target{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button-group .bp3-button.bp3-fill,
  .bp3-button-group.bp3-fill .bp3-button:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-vertical{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    vertical-align:top; }
    .bp3-button-group.bp3-vertical.bp3-fill{
      width:unset;
      height:100%; }
    .bp3-button-group.bp3-vertical .bp3-button{
      margin-right:0 !important;
      width:100%; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:first-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:first-child{
      border-radius:3px 3px 0 0; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:last-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:last-child{
      border-radius:0 0 3px 3px; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:not(:last-child){
      margin-bottom:-1px; }
  .bp3-button-group.bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    margin-right:1px; }
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-button:not(:last-child){
    margin-bottom:1px; }
.bp3-callout{
  line-height:1.5;
  font-size:14px;
  position:relative;
  border-radius:3px;
  background-color:rgba(138, 155, 168, 0.15);
  width:100%;
  padding:10px 12px 9px; }
  .bp3-callout[class*="bp3-icon-"]{
    padding-left:40px; }
    .bp3-callout[class*="bp3-icon-"]::before{
      line-height:1;
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-weight:400;
      font-style:normal;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      position:absolute;
      top:10px;
      left:10px;
      color:#5c7080; }
  .bp3-callout.bp3-callout-icon{
    padding-left:40px; }
    .bp3-callout.bp3-callout-icon > .bp3-icon:first-child{
      position:absolute;
      top:10px;
      left:10px;
      color:#5c7080; }
  .bp3-callout .bp3-heading{
    margin-top:0;
    margin-bottom:5px;
    line-height:20px; }
    .bp3-callout .bp3-heading:last-child{
      margin-bottom:0; }
  .bp3-dark .bp3-callout{
    background-color:rgba(138, 155, 168, 0.2); }
    .bp3-dark .bp3-callout[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
  .bp3-callout.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15); }
    .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-primary .bp3-heading{
      color:#106ba3; }
    .bp3-dark .bp3-callout.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-primary .bp3-heading{
        color:#48aff0; }
  .bp3-callout.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15); }
    .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-success .bp3-heading{
      color:#0d8050; }
    .bp3-dark .bp3-callout.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-success .bp3-heading{
        color:#3dcc91; }
  .bp3-callout.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15); }
    .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-warning .bp3-heading{
      color:#bf7326; }
    .bp3-dark .bp3-callout.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-warning .bp3-heading{
        color:#ffb366; }
  .bp3-callout.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15); }
    .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-danger .bp3-heading{
      color:#c23030; }
    .bp3-dark .bp3-callout.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-danger .bp3-heading{
        color:#ff7373; }
  .bp3-running-text .bp3-callout{
    margin:20px 0; }
.bp3-card{
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
  background-color:#ffffff;
  padding:20px;
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-card.bp3-dark,
  .bp3-dark .bp3-card{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
    background-color:#30404d; }

.bp3-elevation-0{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
  .bp3-elevation-0.bp3-dark,
  .bp3-dark .bp3-elevation-0{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-1{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-1.bp3-dark,
  .bp3-dark .bp3-elevation-1{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-elevation-2{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-2.bp3-dark,
  .bp3-dark .bp3-elevation-2{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4); }

.bp3-elevation-3{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-3.bp3-dark,
  .bp3-dark .bp3-elevation-3{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-elevation-4{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-4.bp3-dark,
  .bp3-dark .bp3-elevation-4{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:hover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  cursor:pointer; }
  .bp3-card.bp3-interactive:hover.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:active{
  opacity:0.9;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  -webkit-transition-duration:0;
          transition-duration:0; }
  .bp3-card.bp3-interactive:active.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-collapse{
  height:0;
  overflow-y:hidden;
  -webkit-transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-collapse .bp3-collapse-body{
    -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-collapse .bp3-collapse-body[aria-hidden="true"]{
      display:none; }

.bp3-context-menu .bp3-popover-target{
  display:block; }

.bp3-context-menu-popover-target{
  position:fixed; }

.bp3-divider{
  margin:5px;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  border-bottom:1px solid rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-dialog-container{
  opacity:1;
  -webkit-transform:scale(1);
          transform:scale(1);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  width:100%;
  min-height:100%;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-dialog-container.bp3-overlay-enter > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5); }
  .bp3-dialog-container.bp3-overlay-enter-active > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear-active > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-dialog-container.bp3-overlay-exit > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-dialog-container.bp3-overlay-exit-active > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5);
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }

.bp3-dialog{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:30px 0;
  border-radius:6px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  background:#ebf1f5;
  width:500px;
  padding-bottom:20px;
  pointer-events:all;
  -webkit-user-select:text;
     -moz-user-select:text;
      -ms-user-select:text;
          user-select:text; }
  .bp3-dialog:focus{
    outline:0; }
  .bp3-dialog.bp3-dark,
  .bp3-dark .bp3-dialog{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    background:#293742;
    color:#f5f8fa; }

.bp3-dialog-header{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border-radius:6px 6px 0 0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  background:#ffffff;
  min-height:40px;
  padding-right:5px;
  padding-left:20px; }
  .bp3-dialog-header .bp3-icon-large,
  .bp3-dialog-header .bp3-icon{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px;
    color:#5c7080; }
  .bp3-dialog-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    margin:0;
    line-height:inherit; }
    .bp3-dialog-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-dialog-header{
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
    background:#30404d; }
    .bp3-dark .bp3-dialog-header .bp3-icon-large,
    .bp3-dark .bp3-dialog-header .bp3-icon{
      color:#a7b6c2; }

.bp3-dialog-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  margin:20px;
  line-height:18px; }

.bp3-dialog-footer{
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  margin:0 20px; }

.bp3-dialog-footer-actions{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:end;
      -ms-flex-pack:end;
          justify-content:flex-end; }
  .bp3-dialog-footer-actions .bp3-button{
    margin-left:10px; }
.bp3-drawer{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  background:#ffffff;
  padding:0; }
  .bp3-drawer:focus{
    outline:0; }
  .bp3-drawer.bp3-position-top{
    top:0;
    right:0;
    left:0;
    height:50%; }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter, .bp3-drawer.bp3-position-top.bp3-overlay-appear{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%); }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter-active, .bp3-drawer.bp3-position-top.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit-active{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer.bp3-position-bottom{
    right:0;
    bottom:0;
    left:0;
    height:50%; }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter-active, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer.bp3-position-left{
    top:0;
    bottom:0;
    left:0;
    width:50%; }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter, .bp3-drawer.bp3-position-left.bp3-overlay-appear{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%); }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter-active, .bp3-drawer.bp3-position-left.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit-active{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer.bp3-position-right{
    top:0;
    right:0;
    bottom:0;
    width:50%; }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter, .bp3-drawer.bp3-position-right.bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter-active, .bp3-drawer.bp3-position-right.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right):not(.bp3-vertical){
    top:0;
    right:0;
    bottom:0;
    width:50%; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right).bp3-vertical{
    right:0;
    bottom:0;
    left:0;
    height:50%; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
      -webkit-transition-delay:0;
              transition-delay:0; }
  .bp3-drawer.bp3-dark,
  .bp3-dark .bp3-drawer{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    background:#30404d;
    color:#f5f8fa; }

.bp3-drawer-header{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  position:relative;
  border-radius:0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  min-height:40px;
  padding:5px;
  padding-left:20px; }
  .bp3-drawer-header .bp3-icon-large,
  .bp3-drawer-header .bp3-icon{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px;
    color:#5c7080; }
  .bp3-drawer-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    margin:0;
    line-height:inherit; }
    .bp3-drawer-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-drawer-header{
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-drawer-header .bp3-icon-large,
    .bp3-dark .bp3-drawer-header .bp3-icon{
      color:#a7b6c2; }

.bp3-drawer-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  overflow:auto;
  line-height:18px; }

.bp3-drawer-footer{
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  position:relative;
  -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
  padding:10px 20px; }
  .bp3-dark .bp3-drawer-footer{
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4); }
.bp3-editable-text{
  display:inline-block;
  position:relative;
  cursor:text;
  max-width:100%;
  vertical-align:top;
  white-space:nowrap; }
  .bp3-editable-text::before{
    position:absolute;
    top:-3px;
    right:-3px;
    bottom:-3px;
    left:-3px;
    border-radius:3px;
    content:"";
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-editable-text.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
    background-color:#ffffff; }
  .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#137cbd; }
  .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4); }
  .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#0f9960; }
  .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4); }
  .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#d9822b; }
  .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4); }
  .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#db3737; }
  .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4); }
  .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15); }
  .bp3-dark .bp3-editable-text.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background-color:rgba(16, 22, 26, 0.3); }
  .bp3-dark .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#48aff0; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4);
            box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#3dcc91; }
  .bp3-dark .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4);
            box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#ffb366; }
  .bp3-dark .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4);
            box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#ff7373; }
  .bp3-dark .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4);
            box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-editable-text-input,
.bp3-editable-text-content{
  display:inherit;
  position:relative;
  min-width:inherit;
  max-width:inherit;
  vertical-align:top;
  text-transform:inherit;
  letter-spacing:inherit;
  color:inherit;
  font:inherit;
  resize:none; }

.bp3-editable-text-input{
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  background:none;
  width:100%;
  padding:0;
  white-space:pre-wrap; }
  .bp3-editable-text-input::-webkit-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input::-moz-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input:-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input::-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input::placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-editable-text-input:focus{
    outline:none; }
  .bp3-editable-text-input::-ms-clear{
    display:none; }

.bp3-editable-text-content{
  overflow:hidden;
  padding-right:2px;
  text-overflow:ellipsis;
  white-space:pre; }
  .bp3-editable-text-editing > .bp3-editable-text-content{
    position:absolute;
    left:0;
    visibility:hidden; }
  .bp3-editable-text-placeholder > .bp3-editable-text-content{
    color:rgba(92, 112, 128, 0.6); }
    .bp3-dark .bp3-editable-text-placeholder > .bp3-editable-text-content{
      color:rgba(167, 182, 194, 0.6); }

.bp3-editable-text.bp3-multiline{
  display:block; }
  .bp3-editable-text.bp3-multiline .bp3-editable-text-content{
    overflow:auto;
    white-space:pre-wrap;
    word-wrap:break-word; }
.bp3-control-group{
  -webkit-transform:translateZ(0);
          transform:translateZ(0);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:stretch;
      -ms-flex-align:stretch;
          align-items:stretch; }
  .bp3-control-group > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select,
  .bp3-control-group .bp3-input,
  .bp3-control-group .bp3-select{
    position:relative; }
  .bp3-control-group .bp3-input{
    z-index:2;
    border-radius:inherit; }
    .bp3-control-group .bp3-input:focus{
      z-index:14;
      border-radius:3px; }
    .bp3-control-group .bp3-input[class*="bp3-intent"]{
      z-index:13; }
      .bp3-control-group .bp3-input[class*="bp3-intent"]:focus{
        z-index:15; }
    .bp3-control-group .bp3-input[readonly], .bp3-control-group .bp3-input:disabled, .bp3-control-group .bp3-input.bp3-disabled{
      z-index:1; }
  .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input{
    z-index:13; }
    .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input:focus{
      z-index:15; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select select,
  .bp3-control-group .bp3-select select{
    -webkit-transform:translateZ(0);
            transform:translateZ(0);
    z-index:4;
    border-radius:inherit; }
    .bp3-control-group .bp3-button:focus,
    .bp3-control-group .bp3-html-select select:focus,
    .bp3-control-group .bp3-select select:focus{
      z-index:5; }
    .bp3-control-group .bp3-button:hover,
    .bp3-control-group .bp3-html-select select:hover,
    .bp3-control-group .bp3-select select:hover{
      z-index:6; }
    .bp3-control-group .bp3-button:active,
    .bp3-control-group .bp3-html-select select:active,
    .bp3-control-group .bp3-select select:active{
      z-index:7; }
    .bp3-control-group .bp3-button[readonly], .bp3-control-group .bp3-button:disabled, .bp3-control-group .bp3-button.bp3-disabled,
    .bp3-control-group .bp3-html-select select[readonly],
    .bp3-control-group .bp3-html-select select:disabled,
    .bp3-control-group .bp3-html-select select.bp3-disabled,
    .bp3-control-group .bp3-select select[readonly],
    .bp3-control-group .bp3-select select:disabled,
    .bp3-control-group .bp3-select select.bp3-disabled{
      z-index:3; }
    .bp3-control-group .bp3-button[class*="bp3-intent"],
    .bp3-control-group .bp3-html-select select[class*="bp3-intent"],
    .bp3-control-group .bp3-select select[class*="bp3-intent"]{
      z-index:9; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:focus{
        z-index:10; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:hover{
        z-index:11; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:active{
        z-index:12; }
      .bp3-control-group .bp3-button[class*="bp3-intent"][readonly], .bp3-control-group .bp3-button[class*="bp3-intent"]:disabled, .bp3-control-group .bp3-button[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"].bp3-disabled{
        z-index:8; }
  .bp3-control-group .bp3-input-group > .bp3-icon,
  .bp3-control-group .bp3-input-group > .bp3-button,
  .bp3-control-group .bp3-input-group > .bp3-input-action{
    z-index:16; }
  .bp3-control-group .bp3-select::after,
  .bp3-control-group .bp3-html-select::after,
  .bp3-control-group .bp3-select > .bp3-icon,
  .bp3-control-group .bp3-html-select > .bp3-icon{
    z-index:17; }
  .bp3-control-group:not(.bp3-vertical) > *{
    margin-right:-1px; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > *{
    margin-right:0; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > .bp3-button + .bp3-button{
    margin-left:1px; }
  .bp3-control-group .bp3-popover-wrapper,
  .bp3-control-group .bp3-popover-target{
    border-radius:inherit; }
  .bp3-control-group > :first-child{
    border-radius:3px 0 0 3px; }
  .bp3-control-group > :last-child{
    margin-right:0;
    border-radius:0 3px 3px 0; }
  .bp3-control-group > :only-child{
    margin-right:0;
    border-radius:3px; }
  .bp3-control-group .bp3-input-group .bp3-button{
    border-radius:3px; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-fill > *:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-vertical{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-control-group.bp3-vertical > *{
      margin-top:-1px; }
    .bp3-control-group.bp3-vertical > :first-child{
      margin-top:0;
      border-radius:3px 3px 0 0; }
    .bp3-control-group.bp3-vertical > :last-child{
      border-radius:0 0 3px 3px; }
.bp3-control{
  display:block;
  position:relative;
  margin-bottom:10px;
  cursor:pointer;
  text-transform:none; }
  .bp3-control input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
  .bp3-control:hover input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#106ba3; }
  .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background:#0e5a8a; }
  .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(19, 124, 189, 0.5); }
  .bp3-dark .bp3-control input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control:hover input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#106ba3; }
  .bp3-dark .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#0e5a8a; }
  .bp3-dark .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(14, 90, 138, 0.5); }
  .bp3-control:not(.bp3-align-right){
    padding-left:26px; }
    .bp3-control:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-26px; }
  .bp3-control.bp3-align-right{
    padding-right:26px; }
    .bp3-control.bp3-align-right .bp3-control-indicator{
      margin-right:-26px; }
  .bp3-control.bp3-disabled{
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-control.bp3-inline{
    display:inline-block;
    margin-right:20px; }
  .bp3-control input{
    position:absolute;
    top:0;
    left:0;
    opacity:0;
    z-index:-1; }
  .bp3-control .bp3-control-indicator{
    display:inline-block;
    position:relative;
    margin-top:-3px;
    margin-right:10px;
    border:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    cursor:pointer;
    width:1em;
    height:1em;
    vertical-align:middle;
    font-size:16px;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none; }
    .bp3-control .bp3-control-indicator::before{
      display:block;
      width:1em;
      height:1em;
      content:""; }
  .bp3-control:hover .bp3-control-indicator{
    background-color:#ebf1f5; }
  .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background:#d8e1e8; }
  .bp3-control input:disabled ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(206, 217, 224, 0.5);
    cursor:not-allowed; }
  .bp3-control input:focus ~ .bp3-control-indicator{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:2px;
    -moz-outline-radius:6px; }
  .bp3-control.bp3-align-right .bp3-control-indicator{
    float:right;
    margin-top:1px;
    margin-left:10px; }
  .bp3-control.bp3-large{
    font-size:16px; }
    .bp3-control.bp3-large:not(.bp3-align-right){
      padding-left:30px; }
      .bp3-control.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-30px; }
    .bp3-control.bp3-large.bp3-align-right{
      padding-right:30px; }
      .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
        margin-right:-30px; }
    .bp3-control.bp3-large .bp3-control-indicator{
      font-size:20px; }
    .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-top:0; }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    color:#ffffff; }
  .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    background-color:#106ba3; }
  .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background:#0e5a8a; }
  .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(19, 124, 189, 0.5); }
  .bp3-dark .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#106ba3; }
  .bp3-dark .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#0e5a8a; }
  .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(14, 90, 138, 0.5); }
  .bp3-control.bp3-checkbox .bp3-control-indicator{
    border-radius:3px; }
  .bp3-control.bp3-checkbox input:checked ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M12 5c-.28 0-.53.11-.71.29L7 9.59l-2.29-2.3a1.003 1.003 0 0 0-1.42 1.42l3 3c.18.18.43.29.71.29s.53-.11.71-.29l5-5A1.003 1.003 0 0 0 12 5z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M11 7H5c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-radio .bp3-control-indicator{
    border-radius:50%; }
  .bp3-control.bp3-radio input:checked ~ .bp3-control-indicator::before{
    background-image:radial-gradient(#ffffff, #ffffff 28%, transparent 32%); }
  .bp3-control.bp3-radio input:checked:disabled ~ .bp3-control-indicator::before{
    opacity:0.5; }
  .bp3-control.bp3-radio input:focus ~ .bp3-control-indicator{
    -moz-outline-radius:16px; }
  .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(167, 182, 194, 0.5); }
  .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(115, 134, 148, 0.5); }
  .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(92, 112, 128, 0.5); }
  .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5); }
    .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5); }
    .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch:not(.bp3-align-right){
    padding-left:38px; }
    .bp3-control.bp3-switch:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-38px; }
  .bp3-control.bp3-switch.bp3-align-right{
    padding-right:38px; }
    .bp3-control.bp3-switch.bp3-align-right .bp3-control-indicator{
      margin-right:-38px; }
  .bp3-control.bp3-switch .bp3-control-indicator{
    border:none;
    border-radius:1.75em;
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    width:auto;
    min-width:1.75em;
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-control.bp3-switch .bp3-control-indicator::before{
      position:absolute;
      left:0;
      margin:2px;
      border-radius:50%;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
      background:#ffffff;
      width:calc(1em - 4px);
      height:calc(1em - 4px);
      -webkit-transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    left:calc(100% - 1em); }
  .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right){
    padding-left:45px; }
    .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-45px; }
  .bp3-control.bp3-switch.bp3-large.bp3-align-right{
    padding-right:45px; }
    .bp3-control.bp3-switch.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-right:-45px; }
  .bp3-dark .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.7); }
  .bp3-dark .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.9); }
  .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(57, 75, 89, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-dark .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-dark .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch .bp3-control-indicator::before{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background:#394b59; }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-control.bp3-switch .bp3-switch-inner-text{
    text-align:center;
    font-size:0.7em; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:first-child{
    visibility:hidden;
    margin-right:1.2em;
    margin-left:0.5em;
    line-height:0; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:last-child{
    visibility:visible;
    margin-right:0.5em;
    margin-left:1.2em;
    line-height:1em; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:first-child{
    visibility:visible;
    line-height:1em; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:last-child{
    visibility:hidden;
    line-height:0; }
  .bp3-dark .bp3-control{
    color:#f5f8fa; }
    .bp3-dark .bp3-control.bp3-disabled{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-control .bp3-control-indicator{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0)); }
    .bp3-dark .bp3-control:hover .bp3-control-indicator{
      background-color:#30404d; }
    .bp3-dark .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background:#202b33; }
    .bp3-dark .bp3-control input:disabled ~ .bp3-control-indicator{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(57, 75, 89, 0.5);
      cursor:not-allowed; }
    .bp3-dark .bp3-control.bp3-checkbox input:disabled:checked ~ .bp3-control-indicator, .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
      color:rgba(167, 182, 194, 0.6); }
.bp3-file-input{
  display:inline-block;
  position:relative;
  cursor:pointer;
  height:30px; }
  .bp3-file-input input{
    opacity:0;
    margin:0;
    min-width:200px; }
    .bp3-file-input input:disabled + .bp3-file-upload-input,
    .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(206, 217, 224, 0.5);
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6);
      resize:none; }
      .bp3-file-input input:disabled + .bp3-file-upload-input::after,
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
        outline:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        background-color:rgba(206, 217, 224, 0.5);
        background-image:none;
        cursor:not-allowed;
        color:rgba(92, 112, 128, 0.6); }
        .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active:hover,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active:hover{
          background:rgba(206, 217, 224, 0.7); }
      .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input, .bp3-dark
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
        -webkit-box-shadow:none;
                box-shadow:none;
        background:rgba(57, 75, 89, 0.5);
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after, .bp3-dark
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
          -webkit-box-shadow:none;
                  box-shadow:none;
          background-color:rgba(57, 75, 89, 0.5);
          background-image:none;
          color:rgba(167, 182, 194, 0.6); }
          .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-dark
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active{
            background:rgba(57, 75, 89, 0.7); }
  .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#182026; }
  .bp3-dark .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#f5f8fa; }
  .bp3-file-input.bp3-fill{
    width:100%; }
  .bp3-file-input.bp3-large,
  .bp3-large .bp3-file-input{
    height:40px; }
  .bp3-file-input .bp3-file-upload-input-custom-text::after{
    content:attr(bp3-button-text); }

.bp3-file-upload-input{
  outline:none;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  background:#ffffff;
  height:30px;
  padding:0 10px;
  vertical-align:middle;
  line-height:30px;
  color:#182026;
  font-size:14px;
  font-weight:400;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  position:absolute;
  top:0;
  right:0;
  left:0;
  padding-right:80px;
  color:rgba(92, 112, 128, 0.6);
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-file-upload-input::-webkit-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input::-moz-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input:-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input::-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input::placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-file-upload-input:focus, .bp3-file-upload-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-file-upload-input[type="search"], .bp3-file-upload-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-file-upload-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-file-upload-input:disabled, .bp3-file-upload-input.bp3-disabled{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(206, 217, 224, 0.5);
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6);
    resize:none; }
  .bp3-file-upload-input::after{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    color:#182026;
    min-width:24px;
    min-height:24px;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    position:absolute;
    top:0;
    right:0;
    margin:3px;
    border-radius:3px;
    width:70px;
    text-align:center;
    line-height:24px;
    content:"Browse"; }
    .bp3-file-upload-input::after:hover{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
      background-clip:padding-box;
      background-color:#ebf1f5; }
    .bp3-file-upload-input::after:active, .bp3-file-upload-input::after.bp3-active{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#d8e1e8;
      background-image:none; }
    .bp3-file-upload-input::after:disabled, .bp3-file-upload-input::after.bp3-disabled{
      outline:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      cursor:not-allowed;
      color:rgba(92, 112, 128, 0.6); }
      .bp3-file-upload-input::after:disabled.bp3-active, .bp3-file-upload-input::after:disabled.bp3-active:hover, .bp3-file-upload-input::after.bp3-disabled.bp3-active, .bp3-file-upload-input::after.bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-file-upload-input:hover::after{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#ebf1f5; }
  .bp3-file-upload-input:active::after{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#d8e1e8;
    background-image:none; }
  .bp3-large .bp3-file-upload-input{
    height:40px;
    line-height:40px;
    font-size:16px;
    padding-right:95px; }
    .bp3-large .bp3-file-upload-input[type="search"], .bp3-large .bp3-file-upload-input.bp3-round{
      padding:0 15px; }
    .bp3-large .bp3-file-upload-input::after{
      min-width:30px;
      min-height:30px;
      margin:5px;
      width:85px;
      line-height:30px; }
  .bp3-dark .bp3-file-upload-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#f5f8fa;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:disabled, .bp3-dark .bp3-file-upload-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(57, 75, 89, 0.5);
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::after{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover, .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover{
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
        background-color:#30404d; }
      .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
        background-color:#202b33;
        background-image:none; }
      .bp3-dark .bp3-file-upload-input::after:disabled, .bp3-dark .bp3-file-upload-input::after.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none;
        background-color:rgba(57, 75, 89, 0.5);
        background-image:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-upload-input::after:disabled.bp3-active, .bp3-dark .bp3-file-upload-input::after.bp3-disabled.bp3-active{
          background:rgba(57, 75, 89, 0.7); }
      .bp3-dark .bp3-file-upload-input::after .bp3-button-spinner .bp3-spinner-head{
        background:rgba(16, 22, 26, 0.5);
        stroke:#8a9ba8; }
    .bp3-dark .bp3-file-upload-input:hover::after{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#30404d; }
    .bp3-dark .bp3-file-upload-input:active::after{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#202b33;
      background-image:none; }

.bp3-file-upload-input::after{
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
.bp3-form-group{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0 0 15px; }
  .bp3-form-group label.bp3-label{
    margin-bottom:5px; }
  .bp3-form-group .bp3-control{
    margin-top:7px; }
  .bp3-form-group .bp3-form-helper-text{
    margin-top:5px;
    color:#5c7080;
    font-size:12px; }
  .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#106ba3; }
  .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#0d8050; }
  .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#bf7326; }
  .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#c23030; }
  .bp3-form-group.bp3-inline{
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
    .bp3-form-group.bp3-inline.bp3-large label.bp3-label{
      margin:0 10px 0 0;
      line-height:40px; }
    .bp3-form-group.bp3-inline label.bp3-label{
      margin:0 10px 0 0;
      line-height:30px; }
  .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-dark .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#48aff0; }
  .bp3-dark .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#3dcc91; }
  .bp3-dark .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#ffb366; }
  .bp3-dark .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#ff7373; }
  .bp3-dark .bp3-form-group .bp3-form-helper-text{
    color:#a7b6c2; }
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(167, 182, 194, 0.6) !important; }
.bp3-input-group{
  display:block;
  position:relative; }
  .bp3-input-group .bp3-input{
    position:relative;
    width:100%; }
    .bp3-input-group .bp3-input:not(:first-child){
      padding-left:30px; }
    .bp3-input-group .bp3-input:not(:last-child){
      padding-right:30px; }
  .bp3-input-group .bp3-input-action,
  .bp3-input-group > .bp3-button,
  .bp3-input-group > .bp3-icon{
    position:absolute;
    top:0; }
    .bp3-input-group .bp3-input-action:first-child,
    .bp3-input-group > .bp3-button:first-child,
    .bp3-input-group > .bp3-icon:first-child{
      left:0; }
    .bp3-input-group .bp3-input-action:last-child,
    .bp3-input-group > .bp3-button:last-child,
    .bp3-input-group > .bp3-icon:last-child{
      right:0; }
  .bp3-input-group .bp3-button{
    min-width:24px;
    min-height:24px;
    margin:3px;
    padding:0 7px; }
    .bp3-input-group .bp3-button:empty{
      padding:0; }
  .bp3-input-group > .bp3-icon{
    z-index:1;
    color:#5c7080; }
    .bp3-input-group > .bp3-icon:empty{
      line-height:1;
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-weight:400;
      font-style:normal;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased; }
  .bp3-input-group > .bp3-icon,
  .bp3-input-group .bp3-input-action > .bp3-spinner{
    margin:7px; }
  .bp3-input-group .bp3-tag{
    margin:5px; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus),
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
    color:#5c7080; }
    .bp3-dark .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus), .bp3-dark
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
      color:#a7b6c2; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large{
      color:#5c7080; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled,
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled{
    color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-large{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-input-group.bp3-disabled{
    cursor:not-allowed; }
    .bp3-input-group.bp3-disabled .bp3-icon{
      color:rgba(92, 112, 128, 0.6); }
  .bp3-input-group.bp3-large .bp3-button{
    min-width:30px;
    min-height:30px;
    margin:5px; }
  .bp3-input-group.bp3-large > .bp3-icon,
  .bp3-input-group.bp3-large .bp3-input-action > .bp3-spinner{
    margin:12px; }
  .bp3-input-group.bp3-large .bp3-input{
    height:40px;
    line-height:40px;
    font-size:16px; }
    .bp3-input-group.bp3-large .bp3-input[type="search"], .bp3-input-group.bp3-large .bp3-input.bp3-round{
      padding:0 15px; }
    .bp3-input-group.bp3-large .bp3-input:not(:first-child){
      padding-left:40px; }
    .bp3-input-group.bp3-large .bp3-input:not(:last-child){
      padding-right:40px; }
  .bp3-input-group.bp3-small .bp3-button{
    min-width:20px;
    min-height:20px;
    margin:2px; }
  .bp3-input-group.bp3-small .bp3-tag{
    min-width:20px;
    min-height:20px;
    margin:2px; }
  .bp3-input-group.bp3-small > .bp3-icon,
  .bp3-input-group.bp3-small .bp3-input-action > .bp3-spinner{
    margin:4px; }
  .bp3-input-group.bp3-small .bp3-input{
    height:24px;
    padding-right:8px;
    padding-left:8px;
    line-height:24px;
    font-size:12px; }
    .bp3-input-group.bp3-small .bp3-input[type="search"], .bp3-input-group.bp3-small .bp3-input.bp3-round{
      padding:0 12px; }
    .bp3-input-group.bp3-small .bp3-input:not(:first-child){
      padding-left:24px; }
    .bp3-input-group.bp3-small .bp3-input:not(:last-child){
      padding-right:24px; }
  .bp3-input-group.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-input-group.bp3-round .bp3-button,
  .bp3-input-group.bp3-round .bp3-input,
  .bp3-input-group.bp3-round .bp3-tag{
    border-radius:30px; }
  .bp3-dark .bp3-input-group .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-input-group.bp3-disabled .bp3-icon{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-input-group.bp3-intent-primary .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input-group.bp3-intent-primary .bp3-input:disabled, .bp3-input-group.bp3-intent-primary .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-primary > .bp3-icon{
    color:#106ba3; }
    .bp3-dark .bp3-input-group.bp3-intent-primary > .bp3-icon{
      color:#48aff0; }
  .bp3-input-group.bp3-intent-success .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input-group.bp3-intent-success .bp3-input:disabled, .bp3-input-group.bp3-intent-success .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-success > .bp3-icon{
    color:#0d8050; }
    .bp3-dark .bp3-input-group.bp3-intent-success > .bp3-icon{
      color:#3dcc91; }
  .bp3-input-group.bp3-intent-warning .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input-group.bp3-intent-warning .bp3-input:disabled, .bp3-input-group.bp3-intent-warning .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-warning > .bp3-icon{
    color:#bf7326; }
    .bp3-dark .bp3-input-group.bp3-intent-warning > .bp3-icon{
      color:#ffb366; }
  .bp3-input-group.bp3-intent-danger .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input-group.bp3-intent-danger .bp3-input:disabled, .bp3-input-group.bp3-intent-danger .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-danger > .bp3-icon{
    color:#c23030; }
    .bp3-dark .bp3-input-group.bp3-intent-danger > .bp3-icon{
      color:#ff7373; }
.bp3-input{
  outline:none;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  background:#ffffff;
  height:30px;
  padding:0 10px;
  vertical-align:middle;
  line-height:30px;
  color:#182026;
  font-size:14px;
  font-weight:400;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none; }
  .bp3-input::-webkit-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input::-moz-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input:-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input::-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input::placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input:focus, .bp3-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-input[type="search"], .bp3-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-input:disabled, .bp3-input.bp3-disabled{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(206, 217, 224, 0.5);
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6);
    resize:none; }
  .bp3-input.bp3-large{
    height:40px;
    line-height:40px;
    font-size:16px; }
    .bp3-input.bp3-large[type="search"], .bp3-input.bp3-large.bp3-round{
      padding:0 15px; }
  .bp3-input.bp3-small{
    height:24px;
    padding-right:8px;
    padding-left:8px;
    line-height:24px;
    font-size:12px; }
    .bp3-input.bp3-small[type="search"], .bp3-input.bp3-small.bp3-round{
      padding:0 12px; }
  .bp3-input.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-dark .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#f5f8fa; }
    .bp3-dark .bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input:disabled, .bp3-dark .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(57, 75, 89, 0.5);
      color:rgba(167, 182, 194, 0.6); }
  .bp3-input.bp3-intent-primary{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input.bp3-intent-primary:disabled, .bp3-input.bp3-intent-primary.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                box-shadow:inset 0 0 0 1px #137cbd; }
      .bp3-dark .bp3-input.bp3-intent-primary:disabled, .bp3-dark .bp3-input.bp3-intent-primary.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-success{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input.bp3-intent-success:disabled, .bp3-input.bp3-intent-success.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-success{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success:focus{
        -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                box-shadow:inset 0 0 0 1px #0f9960; }
      .bp3-dark .bp3-input.bp3-intent-success:disabled, .bp3-dark .bp3-input.bp3-intent-success.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-warning{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input.bp3-intent-warning:disabled, .bp3-input.bp3-intent-warning.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning:focus{
        -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                box-shadow:inset 0 0 0 1px #d9822b; }
      .bp3-dark .bp3-input.bp3-intent-warning:disabled, .bp3-dark .bp3-input.bp3-intent-warning.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-danger{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input.bp3-intent-danger:disabled, .bp3-input.bp3-intent-danger.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger:focus{
        -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #db3737;
                box-shadow:inset 0 0 0 1px #db3737; }
      .bp3-dark .bp3-input.bp3-intent-danger:disabled, .bp3-dark .bp3-input.bp3-intent-danger.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input::-ms-clear{
    display:none; }
textarea.bp3-input{
  max-width:100%;
  padding:10px; }
  textarea.bp3-input, textarea.bp3-input.bp3-large, textarea.bp3-input.bp3-small{
    height:auto;
    line-height:inherit; }
  textarea.bp3-input.bp3-small{
    padding:8px; }
  .bp3-dark textarea.bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background:rgba(16, 22, 26, 0.3);
    color:#f5f8fa; }
    .bp3-dark textarea.bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input:disabled, .bp3-dark textarea.bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:rgba(57, 75, 89, 0.5);
      color:rgba(167, 182, 194, 0.6); }
label.bp3-label{
  display:block;
  margin-top:0;
  margin-bottom:15px; }
  label.bp3-label .bp3-html-select,
  label.bp3-label .bp3-input,
  label.bp3-label .bp3-select,
  label.bp3-label .bp3-slider,
  label.bp3-label .bp3-popover-wrapper{
    display:block;
    margin-top:5px;
    text-transform:none; }
  label.bp3-label .bp3-button-group{
    margin-top:5px; }
  label.bp3-label .bp3-select select,
  label.bp3-label .bp3-html-select select{
    width:100%;
    vertical-align:top;
    font-weight:400; }
  label.bp3-label.bp3-disabled,
  label.bp3-label.bp3-disabled .bp3-text-muted{
    color:rgba(92, 112, 128, 0.6); }
  label.bp3-label.bp3-inline{
    line-height:30px; }
    label.bp3-label.bp3-inline .bp3-html-select,
    label.bp3-label.bp3-inline .bp3-input,
    label.bp3-label.bp3-inline .bp3-input-group,
    label.bp3-label.bp3-inline .bp3-select,
    label.bp3-label.bp3-inline .bp3-popover-wrapper{
      display:inline-block;
      margin:0 0 0 5px;
      vertical-align:top; }
    label.bp3-label.bp3-inline .bp3-button-group{
      margin:0 0 0 5px; }
    label.bp3-label.bp3-inline .bp3-input-group .bp3-input{
      margin-left:0; }
    label.bp3-label.bp3-inline.bp3-large{
      line-height:40px; }
  label.bp3-label:not(.bp3-inline) .bp3-popover-target{
    display:block; }
  .bp3-dark label.bp3-label{
    color:#f5f8fa; }
    .bp3-dark label.bp3-label.bp3-disabled,
    .bp3-dark label.bp3-label.bp3-disabled .bp3-text-muted{
      color:rgba(167, 182, 194, 0.6); }
.bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button{
  -webkit-box-flex:1;
      -ms-flex:1 1 14px;
          flex:1 1 14px;
  width:30px;
  min-height:0;
  padding:0; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:first-child{
    border-radius:0 3px 0 0; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:last-child{
    border-radius:0 0 3px 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:first-child{
  border-radius:3px 0 0 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:last-child{
  border-radius:0 0 0 3px; }

.bp3-numeric-input.bp3-large .bp3-button-group.bp3-vertical > .bp3-button{
  width:40px; }

form{
  display:block; }
.bp3-html-select select,
.bp3-select select{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  padding:5px 10px;
  vertical-align:middle;
  text-align:left;
  font-size:14px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  color:#182026;
  border-radius:3px;
  width:100%;
  height:30px;
  padding:0 25px 0 10px;
  -moz-appearance:none;
  -webkit-appearance:none; }
  .bp3-html-select select > *, .bp3-select select > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-html-select select > .bp3-fill, .bp3-select select > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-html-select select::before,
  .bp3-select select::before, .bp3-html-select select > *, .bp3-select select > *{
    margin-right:7px; }
  .bp3-html-select select:empty::before,
  .bp3-select select:empty::before,
  .bp3-html-select select > :last-child,
  .bp3-select select > :last-child{
    margin-right:0; }
  .bp3-html-select select:hover,
  .bp3-select select:hover{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#ebf1f5; }
  .bp3-html-select select:active,
  .bp3-select select:active, .bp3-html-select select.bp3-active,
  .bp3-select select.bp3-active{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#d8e1e8;
    background-image:none; }
  .bp3-html-select select:disabled,
  .bp3-select select:disabled, .bp3-html-select select.bp3-disabled,
  .bp3-select select.bp3-disabled{
    outline:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
    .bp3-html-select select:disabled.bp3-active,
    .bp3-select select:disabled.bp3-active, .bp3-html-select select:disabled.bp3-active:hover,
    .bp3-select select:disabled.bp3-active:hover, .bp3-html-select select.bp3-disabled.bp3-active,
    .bp3-select select.bp3-disabled.bp3-active, .bp3-html-select select.bp3-disabled.bp3-active:hover,
    .bp3-select select.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }

.bp3-html-select.bp3-minimal select,
.bp3-select.bp3-minimal select{
  -webkit-box-shadow:none;
          box-shadow:none;
  background:none; }
  .bp3-html-select.bp3-minimal select:hover,
  .bp3-select.bp3-minimal select:hover{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(167, 182, 194, 0.3);
    text-decoration:none;
    color:#182026; }
  .bp3-html-select.bp3-minimal select:active,
  .bp3-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal select.bp3-active,
  .bp3-select.bp3-minimal select.bp3-active{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:rgba(115, 134, 148, 0.3);
    color:#182026; }
  .bp3-html-select.bp3-minimal select:disabled,
  .bp3-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal select:disabled:hover,
  .bp3-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal select.bp3-disabled,
  .bp3-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal select.bp3-disabled:hover,
  .bp3-select.bp3-minimal select.bp3-disabled:hover{
    background:none;
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
    .bp3-html-select.bp3-minimal select:disabled.bp3-active,
    .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active{
      background:rgba(115, 134, 148, 0.3); }
  .bp3-dark .bp3-html-select.bp3-minimal select, .bp3-html-select.bp3-minimal .bp3-dark select,
  .bp3-dark .bp3-select.bp3-minimal select, .bp3-select.bp3-minimal .bp3-dark select{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:none;
    color:inherit; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover, .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover{
      background:rgba(138, 155, 168, 0.15); }
    .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:rgba(138, 155, 168, 0.3);
      color:#f5f8fa; }
    .bp3-dark .bp3-html-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal .bp3-dark select:disabled,
    .bp3-dark .bp3-select.bp3-minimal select:disabled, .bp3-select.bp3-minimal .bp3-dark select:disabled, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select:disabled:hover, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover{
      background:none;
      cursor:not-allowed;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-html-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active{
        background:rgba(138, 155, 168, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-primary,
  .bp3-select.bp3-minimal select.bp3-intent-primary{
    color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover{
      background:rgba(19, 124, 189, 0.15);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:rgba(19, 124, 189, 0.3);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled{
      background:none;
      color:rgba(16, 107, 163, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active{
        background:rgba(19, 124, 189, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
      stroke:#106ba3; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary{
      color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.2);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(72, 175, 240, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-success,
  .bp3-select.bp3-minimal select.bp3-intent-success{
    color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover{
      background:rgba(15, 153, 96, 0.15);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:rgba(15, 153, 96, 0.3);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled{
      background:none;
      color:rgba(13, 128, 80, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active{
        background:rgba(15, 153, 96, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
      stroke:#0d8050; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success{
      color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.2);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(61, 204, 145, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-warning,
  .bp3-select.bp3-minimal select.bp3-intent-warning{
    color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover{
      background:rgba(217, 130, 43, 0.15);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:rgba(217, 130, 43, 0.3);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled{
      background:none;
      color:rgba(191, 115, 38, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active{
        background:rgba(217, 130, 43, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
      stroke:#bf7326; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning{
      color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.2);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(255, 179, 102, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-danger,
  .bp3-select.bp3-minimal select.bp3-intent-danger{
    color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      -webkit-box-shadow:none;
              box-shadow:none;
      background:none;
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover{
      background:rgba(219, 55, 55, 0.15);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:rgba(219, 55, 55, 0.3);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled{
      background:none;
      color:rgba(194, 48, 48, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active{
        background:rgba(219, 55, 55, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
      stroke:#c23030; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger{
      color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.2);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(255, 115, 115, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }

.bp3-html-select.bp3-large select,
.bp3-select.bp3-large select{
  height:40px;
  padding-right:35px;
  font-size:16px; }

.bp3-dark .bp3-html-select select, .bp3-dark .bp3-select select{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
  background-color:#394b59;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
  color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover, .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#30404d; }
  .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#202b33;
    background-image:none; }
  .bp3-dark .bp3-html-select select:disabled, .bp3-dark .bp3-select select:disabled, .bp3-dark .bp3-html-select select.bp3-disabled, .bp3-dark .bp3-select select.bp3-disabled{
    -webkit-box-shadow:none;
            box-shadow:none;
    background-color:rgba(57, 75, 89, 0.5);
    background-image:none;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-html-select select:disabled.bp3-active, .bp3-dark .bp3-select select:disabled.bp3-active, .bp3-dark .bp3-html-select select.bp3-disabled.bp3-active, .bp3-dark .bp3-select select.bp3-disabled.bp3-active{
      background:rgba(57, 75, 89, 0.7); }
  .bp3-dark .bp3-html-select select .bp3-button-spinner .bp3-spinner-head, .bp3-dark .bp3-select select .bp3-button-spinner .bp3-spinner-head{
    background:rgba(16, 22, 26, 0.5);
    stroke:#8a9ba8; }

.bp3-html-select select:disabled,
.bp3-select select:disabled{
  -webkit-box-shadow:none;
          box-shadow:none;
  background-color:rgba(206, 217, 224, 0.5);
  cursor:not-allowed;
  color:rgba(92, 112, 128, 0.6); }

.bp3-html-select .bp3-icon,
.bp3-select .bp3-icon, .bp3-select::after{
  position:absolute;
  top:7px;
  right:7px;
  color:#5c7080;
  pointer-events:none; }
  .bp3-html-select .bp3-disabled.bp3-icon,
  .bp3-select .bp3-disabled.bp3-icon, .bp3-disabled.bp3-select::after{
    color:rgba(92, 112, 128, 0.6); }
.bp3-html-select,
.bp3-select{
  display:inline-block;
  position:relative;
  vertical-align:middle;
  letter-spacing:normal; }
  .bp3-html-select select::-ms-expand,
  .bp3-select select::-ms-expand{
    display:none; }
  .bp3-html-select .bp3-icon,
  .bp3-select .bp3-icon{
    color:#5c7080; }
    .bp3-html-select .bp3-icon:hover,
    .bp3-select .bp3-icon:hover{
      color:#182026; }
    .bp3-dark .bp3-html-select .bp3-icon, .bp3-dark
    .bp3-select .bp3-icon{
      color:#a7b6c2; }
      .bp3-dark .bp3-html-select .bp3-icon:hover, .bp3-dark
      .bp3-select .bp3-icon:hover{
        color:#f5f8fa; }
  .bp3-html-select.bp3-large::after,
  .bp3-html-select.bp3-large .bp3-icon,
  .bp3-select.bp3-large::after,
  .bp3-select.bp3-large .bp3-icon{
    top:12px;
    right:12px; }
  .bp3-html-select.bp3-fill,
  .bp3-html-select.bp3-fill select,
  .bp3-select.bp3-fill,
  .bp3-select.bp3-fill select{
    width:100%; }
  .bp3-dark .bp3-html-select option, .bp3-dark
  .bp3-select option{
    background-color:#30404d;
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select::after, .bp3-dark
  .bp3-select::after{
    color:#a7b6c2; }

.bp3-select::after{
  line-height:1;
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-weight:400;
  font-style:normal;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  content:""; }
.bp3-running-text table, table.bp3-html-table{
  border-spacing:0;
  font-size:14px; }
  .bp3-running-text table th, table.bp3-html-table th,
  .bp3-running-text table td,
  table.bp3-html-table td{
    padding:11px;
    vertical-align:top;
    text-align:left; }
  .bp3-running-text table th, table.bp3-html-table th{
    color:#182026;
    font-weight:600; }
  
  .bp3-running-text table td,
  table.bp3-html-table td{
    color:#182026; }
  .bp3-running-text table tbody tr:first-child th, table.bp3-html-table tbody tr:first-child th,
  .bp3-running-text table tbody tr:first-child td,
  table.bp3-html-table tbody tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-running-text table th, .bp3-running-text .bp3-dark table th, .bp3-dark table.bp3-html-table th{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table td, .bp3-running-text .bp3-dark table td, .bp3-dark table.bp3-html-table td{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table tbody tr:first-child th, .bp3-running-text .bp3-dark table tbody tr:first-child th, .bp3-dark table.bp3-html-table tbody tr:first-child th,
  .bp3-dark .bp3-running-text table tbody tr:first-child td,
  .bp3-running-text .bp3-dark table tbody tr:first-child td,
  .bp3-dark table.bp3-html-table tbody tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }

table.bp3-html-table.bp3-html-table-condensed th,
table.bp3-html-table.bp3-html-table-condensed td, table.bp3-html-table.bp3-small th,
table.bp3-html-table.bp3-small td{
  padding-top:6px;
  padding-bottom:6px; }

table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
  background:rgba(191, 204, 214, 0.15); }

table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
  -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered tbody tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child){
    -webkit-box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
  -webkit-box-shadow:none;
          box-shadow:none; }
  table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-interactive tbody tr:hover td{
  background-color:rgba(191, 204, 214, 0.3);
  cursor:pointer; }

table.bp3-html-table.bp3-interactive tbody tr:active td{
  background-color:rgba(191, 204, 214, 0.4); }

.bp3-dark table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
  background:rgba(92, 112, 128, 0.15); }

.bp3-dark table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
  -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }

.bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child){
    -webkit-box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15); }

.bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
  -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:first-child{
    -webkit-box-shadow:none;
            box-shadow:none; }

.bp3-dark table.bp3-html-table.bp3-interactive tbody tr:hover td{
  background-color:rgba(92, 112, 128, 0.3);
  cursor:pointer; }

.bp3-dark table.bp3-html-table.bp3-interactive tbody tr:active td{
  background-color:rgba(92, 112, 128, 0.4); }

.bp3-key-combo{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center; }
  .bp3-key-combo > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-key-combo > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-key-combo::before,
  .bp3-key-combo > *{
    margin-right:5px; }
  .bp3-key-combo:empty::before,
  .bp3-key-combo > :last-child{
    margin-right:0; }

.bp3-hotkey-dialog{
  top:40px;
  padding-bottom:0; }
  .bp3-hotkey-dialog .bp3-dialog-body{
    margin:0;
    padding:0; }
  .bp3-hotkey-dialog .bp3-hotkey-label{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1; }

.bp3-hotkey-column{
  margin:auto;
  max-height:80vh;
  overflow-y:auto;
  padding:30px; }
  .bp3-hotkey-column .bp3-heading{
    margin-bottom:20px; }
    .bp3-hotkey-column .bp3-heading:not(:first-child){
      margin-top:40px; }

.bp3-hotkey{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:justify;
      -ms-flex-pack:justify;
          justify-content:space-between;
  margin-right:0;
  margin-left:0; }
  .bp3-hotkey:not(:last-child){
    margin-bottom:10px; }
.bp3-icon{
  display:inline-block;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  vertical-align:text-bottom; }
  .bp3-icon:not(:empty)::before{
    content:"" !important;
    content:unset !important; }
  .bp3-icon > svg{
    display:block; }
    .bp3-icon > svg:not([fill]){
      fill:currentColor; }

.bp3-icon.bp3-intent-primary, .bp3-icon-standard.bp3-intent-primary, .bp3-icon-large.bp3-intent-primary{
  color:#106ba3; }
  .bp3-dark .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-icon-large.bp3-intent-primary{
    color:#48aff0; }

.bp3-icon.bp3-intent-success, .bp3-icon-standard.bp3-intent-success, .bp3-icon-large.bp3-intent-success{
  color:#0d8050; }
  .bp3-dark .bp3-icon.bp3-intent-success, .bp3-dark .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-icon-large.bp3-intent-success{
    color:#3dcc91; }

.bp3-icon.bp3-intent-warning, .bp3-icon-standard.bp3-intent-warning, .bp3-icon-large.bp3-intent-warning{
  color:#bf7326; }
  .bp3-dark .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-icon-large.bp3-intent-warning{
    color:#ffb366; }

.bp3-icon.bp3-intent-danger, .bp3-icon-standard.bp3-intent-danger, .bp3-icon-large.bp3-intent-danger{
  color:#c23030; }
  .bp3-dark .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-icon-large.bp3-intent-danger{
    color:#ff7373; }

span.bp3-icon-standard{
  line-height:1;
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-weight:400;
  font-style:normal;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon-large{
  line-height:1;
  font-family:"Icons20", sans-serif;
  font-size:20px;
  font-weight:400;
  font-style:normal;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon:empty{
  line-height:1;
  font-family:"Icons20";
  font-size:inherit;
  font-weight:400;
  font-style:normal; }
  span.bp3-icon:empty::before{
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased; }

.bp3-icon-add::before{
  content:""; }

.bp3-icon-add-column-left::before{
  content:""; }

.bp3-icon-add-column-right::before{
  content:""; }

.bp3-icon-add-row-bottom::before{
  content:""; }

.bp3-icon-add-row-top::before{
  content:""; }

.bp3-icon-add-to-artifact::before{
  content:""; }

.bp3-icon-add-to-folder::before{
  content:""; }

.bp3-icon-airplane::before{
  content:""; }

.bp3-icon-align-center::before{
  content:""; }

.bp3-icon-align-justify::before{
  content:""; }

.bp3-icon-align-left::before{
  content:""; }

.bp3-icon-align-right::before{
  content:""; }

.bp3-icon-alignment-bottom::before{
  content:""; }

.bp3-icon-alignment-horizontal-center::before{
  content:""; }

.bp3-icon-alignment-left::before{
  content:""; }

.bp3-icon-alignment-right::before{
  content:""; }

.bp3-icon-alignment-top::before{
  content:""; }

.bp3-icon-alignment-vertical-center::before{
  content:""; }

.bp3-icon-annotation::before{
  content:""; }

.bp3-icon-application::before{
  content:""; }

.bp3-icon-applications::before{
  content:""; }

.bp3-icon-archive::before{
  content:""; }

.bp3-icon-arrow-bottom-left::before{
  content:"↙"; }

.bp3-icon-arrow-bottom-right::before{
  content:"↘"; }

.bp3-icon-arrow-down::before{
  content:"↓"; }

.bp3-icon-arrow-left::before{
  content:"←"; }

.bp3-icon-arrow-right::before{
  content:"→"; }

.bp3-icon-arrow-top-left::before{
  content:"↖"; }

.bp3-icon-arrow-top-right::before{
  content:"↗"; }

.bp3-icon-arrow-up::before{
  content:"↑"; }

.bp3-icon-arrows-horizontal::before{
  content:"↔"; }

.bp3-icon-arrows-vertical::before{
  content:"↕"; }

.bp3-icon-asterisk::before{
  content:"*"; }

.bp3-icon-automatic-updates::before{
  content:""; }

.bp3-icon-badge::before{
  content:""; }

.bp3-icon-ban-circle::before{
  content:""; }

.bp3-icon-bank-account::before{
  content:""; }

.bp3-icon-barcode::before{
  content:""; }

.bp3-icon-blank::before{
  content:""; }

.bp3-icon-blocked-person::before{
  content:""; }

.bp3-icon-bold::before{
  content:""; }

.bp3-icon-book::before{
  content:""; }

.bp3-icon-bookmark::before{
  content:""; }

.bp3-icon-box::before{
  content:""; }

.bp3-icon-briefcase::before{
  content:""; }

.bp3-icon-bring-data::before{
  content:""; }

.bp3-icon-build::before{
  content:""; }

.bp3-icon-calculator::before{
  content:""; }

.bp3-icon-calendar::before{
  content:""; }

.bp3-icon-camera::before{
  content:""; }

.bp3-icon-caret-down::before{
  content:"⌄"; }

.bp3-icon-caret-left::before{
  content:"〈"; }

.bp3-icon-caret-right::before{
  content:"〉"; }

.bp3-icon-caret-up::before{
  content:"⌃"; }

.bp3-icon-cell-tower::before{
  content:""; }

.bp3-icon-changes::before{
  content:""; }

.bp3-icon-chart::before{
  content:""; }

.bp3-icon-chat::before{
  content:""; }

.bp3-icon-chevron-backward::before{
  content:""; }

.bp3-icon-chevron-down::before{
  content:""; }

.bp3-icon-chevron-forward::before{
  content:""; }

.bp3-icon-chevron-left::before{
  content:""; }

.bp3-icon-chevron-right::before{
  content:""; }

.bp3-icon-chevron-up::before{
  content:""; }

.bp3-icon-circle::before{
  content:""; }

.bp3-icon-circle-arrow-down::before{
  content:""; }

.bp3-icon-circle-arrow-left::before{
  content:""; }

.bp3-icon-circle-arrow-right::before{
  content:""; }

.bp3-icon-circle-arrow-up::before{
  content:""; }

.bp3-icon-citation::before{
  content:""; }

.bp3-icon-clean::before{
  content:""; }

.bp3-icon-clipboard::before{
  content:""; }

.bp3-icon-cloud::before{
  content:"☁"; }

.bp3-icon-cloud-download::before{
  content:""; }

.bp3-icon-cloud-upload::before{
  content:""; }

.bp3-icon-code::before{
  content:""; }

.bp3-icon-code-block::before{
  content:""; }

.bp3-icon-cog::before{
  content:""; }

.bp3-icon-collapse-all::before{
  content:""; }

.bp3-icon-column-layout::before{
  content:""; }

.bp3-icon-comment::before{
  content:""; }

.bp3-icon-comparison::before{
  content:""; }

.bp3-icon-compass::before{
  content:""; }

.bp3-icon-compressed::before{
  content:""; }

.bp3-icon-confirm::before{
  content:""; }

.bp3-icon-console::before{
  content:""; }

.bp3-icon-contrast::before{
  content:""; }

.bp3-icon-control::before{
  content:""; }

.bp3-icon-credit-card::before{
  content:""; }

.bp3-icon-cross::before{
  content:"✗"; }

.bp3-icon-crown::before{
  content:""; }

.bp3-icon-cube::before{
  content:""; }

.bp3-icon-cube-add::before{
  content:""; }

.bp3-icon-cube-remove::before{
  content:""; }

.bp3-icon-curved-range-chart::before{
  content:""; }

.bp3-icon-cut::before{
  content:""; }

.bp3-icon-dashboard::before{
  content:""; }

.bp3-icon-data-lineage::before{
  content:""; }

.bp3-icon-database::before{
  content:""; }

.bp3-icon-delete::before{
  content:""; }

.bp3-icon-delta::before{
  content:"Δ"; }

.bp3-icon-derive-column::before{
  content:""; }

.bp3-icon-desktop::before{
  content:""; }

.bp3-icon-diagram-tree::before{
  content:""; }

.bp3-icon-direction-left::before{
  content:""; }

.bp3-icon-direction-right::before{
  content:""; }

.bp3-icon-disable::before{
  content:""; }

.bp3-icon-document::before{
  content:""; }

.bp3-icon-document-open::before{
  content:""; }

.bp3-icon-document-share::before{
  content:""; }

.bp3-icon-dollar::before{
  content:"$"; }

.bp3-icon-dot::before{
  content:"•"; }

.bp3-icon-double-caret-horizontal::before{
  content:""; }

.bp3-icon-double-caret-vertical::before{
  content:""; }

.bp3-icon-double-chevron-down::before{
  content:""; }

.bp3-icon-double-chevron-left::before{
  content:""; }

.bp3-icon-double-chevron-right::before{
  content:""; }

.bp3-icon-double-chevron-up::before{
  content:""; }

.bp3-icon-doughnut-chart::before{
  content:""; }

.bp3-icon-download::before{
  content:""; }

.bp3-icon-drag-handle-horizontal::before{
  content:""; }

.bp3-icon-drag-handle-vertical::before{
  content:""; }

.bp3-icon-draw::before{
  content:""; }

.bp3-icon-drive-time::before{
  content:""; }

.bp3-icon-duplicate::before{
  content:""; }

.bp3-icon-edit::before{
  content:"✎"; }

.bp3-icon-eject::before{
  content:"⏏"; }

.bp3-icon-endorsed::before{
  content:""; }

.bp3-icon-envelope::before{
  content:"✉"; }

.bp3-icon-equals::before{
  content:""; }

.bp3-icon-eraser::before{
  content:""; }

.bp3-icon-error::before{
  content:""; }

.bp3-icon-euro::before{
  content:"€"; }

.bp3-icon-exchange::before{
  content:""; }

.bp3-icon-exclude-row::before{
  content:""; }

.bp3-icon-expand-all::before{
  content:""; }

.bp3-icon-export::before{
  content:""; }

.bp3-icon-eye-off::before{
  content:""; }

.bp3-icon-eye-on::before{
  content:""; }

.bp3-icon-eye-open::before{
  content:""; }

.bp3-icon-fast-backward::before{
  content:""; }

.bp3-icon-fast-forward::before{
  content:""; }

.bp3-icon-feed::before{
  content:""; }

.bp3-icon-feed-subscribed::before{
  content:""; }

.bp3-icon-film::before{
  content:""; }

.bp3-icon-filter::before{
  content:""; }

.bp3-icon-filter-keep::before{
  content:""; }

.bp3-icon-filter-list::before{
  content:""; }

.bp3-icon-filter-open::before{
  content:""; }

.bp3-icon-filter-remove::before{
  content:""; }

.bp3-icon-flag::before{
  content:"⚑"; }

.bp3-icon-flame::before{
  content:""; }

.bp3-icon-flash::before{
  content:""; }

.bp3-icon-floppy-disk::before{
  content:""; }

.bp3-icon-flow-branch::before{
  content:""; }

.bp3-icon-flow-end::before{
  content:""; }

.bp3-icon-flow-linear::before{
  content:""; }

.bp3-icon-flow-review::before{
  content:""; }

.bp3-icon-flow-review-branch::before{
  content:""; }

.bp3-icon-flows::before{
  content:""; }

.bp3-icon-folder-close::before{
  content:""; }

.bp3-icon-folder-new::before{
  content:""; }

.bp3-icon-folder-open::before{
  content:""; }

.bp3-icon-folder-shared::before{
  content:""; }

.bp3-icon-folder-shared-open::before{
  content:""; }

.bp3-icon-follower::before{
  content:""; }

.bp3-icon-following::before{
  content:""; }

.bp3-icon-font::before{
  content:""; }

.bp3-icon-fork::before{
  content:""; }

.bp3-icon-form::before{
  content:""; }

.bp3-icon-full-circle::before{
  content:""; }

.bp3-icon-full-stacked-chart::before{
  content:""; }

.bp3-icon-fullscreen::before{
  content:""; }

.bp3-icon-function::before{
  content:""; }

.bp3-icon-gantt-chart::before{
  content:""; }

.bp3-icon-geolocation::before{
  content:""; }

.bp3-icon-geosearch::before{
  content:""; }

.bp3-icon-git-branch::before{
  content:""; }

.bp3-icon-git-commit::before{
  content:""; }

.bp3-icon-git-merge::before{
  content:""; }

.bp3-icon-git-new-branch::before{
  content:""; }

.bp3-icon-git-pull::before{
  content:""; }

.bp3-icon-git-push::before{
  content:""; }

.bp3-icon-git-repo::before{
  content:""; }

.bp3-icon-glass::before{
  content:""; }

.bp3-icon-globe::before{
  content:""; }

.bp3-icon-globe-network::before{
  content:""; }

.bp3-icon-graph::before{
  content:""; }

.bp3-icon-graph-remove::before{
  content:""; }

.bp3-icon-greater-than::before{
  content:""; }

.bp3-icon-greater-than-or-equal-to::before{
  content:""; }

.bp3-icon-grid::before{
  content:""; }

.bp3-icon-grid-view::before{
  content:""; }

.bp3-icon-group-objects::before{
  content:""; }

.bp3-icon-grouped-bar-chart::before{
  content:""; }

.bp3-icon-hand::before{
  content:""; }

.bp3-icon-hand-down::before{
  content:""; }

.bp3-icon-hand-left::before{
  content:""; }

.bp3-icon-hand-right::before{
  content:""; }

.bp3-icon-hand-up::before{
  content:""; }

.bp3-icon-header::before{
  content:""; }

.bp3-icon-header-one::before{
  content:""; }

.bp3-icon-header-two::before{
  content:""; }

.bp3-icon-headset::before{
  content:""; }

.bp3-icon-heart::before{
  content:"♥"; }

.bp3-icon-heart-broken::before{
  content:""; }

.bp3-icon-heat-grid::before{
  content:""; }

.bp3-icon-heatmap::before{
  content:""; }

.bp3-icon-help::before{
  content:"?"; }

.bp3-icon-helper-management::before{
  content:""; }

.bp3-icon-highlight::before{
  content:""; }

.bp3-icon-history::before{
  content:""; }

.bp3-icon-home::before{
  content:"⌂"; }

.bp3-icon-horizontal-bar-chart::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-asc::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-desc::before{
  content:""; }

.bp3-icon-horizontal-distribution::before{
  content:""; }

.bp3-icon-id-number::before{
  content:""; }

.bp3-icon-image-rotate-left::before{
  content:""; }

.bp3-icon-image-rotate-right::before{
  content:""; }

.bp3-icon-import::before{
  content:""; }

.bp3-icon-inbox::before{
  content:""; }

.bp3-icon-inbox-filtered::before{
  content:""; }

.bp3-icon-inbox-geo::before{
  content:""; }

.bp3-icon-inbox-search::before{
  content:""; }

.bp3-icon-inbox-update::before{
  content:""; }

.bp3-icon-info-sign::before{
  content:"ℹ"; }

.bp3-icon-inheritance::before{
  content:""; }

.bp3-icon-inner-join::before{
  content:""; }

.bp3-icon-insert::before{
  content:""; }

.bp3-icon-intersection::before{
  content:""; }

.bp3-icon-ip-address::before{
  content:""; }

.bp3-icon-issue::before{
  content:""; }

.bp3-icon-issue-closed::before{
  content:""; }

.bp3-icon-issue-new::before{
  content:""; }

.bp3-icon-italic::before{
  content:""; }

.bp3-icon-join-table::before{
  content:""; }

.bp3-icon-key::before{
  content:""; }

.bp3-icon-key-backspace::before{
  content:""; }

.bp3-icon-key-command::before{
  content:""; }

.bp3-icon-key-control::before{
  content:""; }

.bp3-icon-key-delete::before{
  content:""; }

.bp3-icon-key-enter::before{
  content:""; }

.bp3-icon-key-escape::before{
  content:""; }

.bp3-icon-key-option::before{
  content:""; }

.bp3-icon-key-shift::before{
  content:""; }

.bp3-icon-key-tab::before{
  content:""; }

.bp3-icon-known-vehicle::before{
  content:""; }

.bp3-icon-label::before{
  content:""; }

.bp3-icon-layer::before{
  content:""; }

.bp3-icon-layers::before{
  content:""; }

.bp3-icon-layout::before{
  content:""; }

.bp3-icon-layout-auto::before{
  content:""; }

.bp3-icon-layout-balloon::before{
  content:""; }

.bp3-icon-layout-circle::before{
  content:""; }

.bp3-icon-layout-grid::before{
  content:""; }

.bp3-icon-layout-group-by::before{
  content:""; }

.bp3-icon-layout-hierarchy::before{
  content:""; }

.bp3-icon-layout-linear::before{
  content:""; }

.bp3-icon-layout-skew-grid::before{
  content:""; }

.bp3-icon-layout-sorted-clusters::before{
  content:""; }

.bp3-icon-learning::before{
  content:""; }

.bp3-icon-left-join::before{
  content:""; }

.bp3-icon-less-than::before{
  content:""; }

.bp3-icon-less-than-or-equal-to::before{
  content:""; }

.bp3-icon-lifesaver::before{
  content:""; }

.bp3-icon-lightbulb::before{
  content:""; }

.bp3-icon-link::before{
  content:""; }

.bp3-icon-list::before{
  content:"☰"; }

.bp3-icon-list-columns::before{
  content:""; }

.bp3-icon-list-detail-view::before{
  content:""; }

.bp3-icon-locate::before{
  content:""; }

.bp3-icon-lock::before{
  content:""; }

.bp3-icon-log-in::before{
  content:""; }

.bp3-icon-log-out::before{
  content:""; }

.bp3-icon-manual::before{
  content:""; }

.bp3-icon-manually-entered-data::before{
  content:""; }

.bp3-icon-map::before{
  content:""; }

.bp3-icon-map-create::before{
  content:""; }

.bp3-icon-map-marker::before{
  content:""; }

.bp3-icon-maximize::before{
  content:""; }

.bp3-icon-media::before{
  content:""; }

.bp3-icon-menu::before{
  content:""; }

.bp3-icon-menu-closed::before{
  content:""; }

.bp3-icon-menu-open::before{
  content:""; }

.bp3-icon-merge-columns::before{
  content:""; }

.bp3-icon-merge-links::before{
  content:""; }

.bp3-icon-minimize::before{
  content:""; }

.bp3-icon-minus::before{
  content:"−"; }

.bp3-icon-mobile-phone::before{
  content:""; }

.bp3-icon-mobile-video::before{
  content:""; }

.bp3-icon-moon::before{
  content:""; }

.bp3-icon-more::before{
  content:""; }

.bp3-icon-mountain::before{
  content:""; }

.bp3-icon-move::before{
  content:""; }

.bp3-icon-mugshot::before{
  content:""; }

.bp3-icon-multi-select::before{
  content:""; }

.bp3-icon-music::before{
  content:""; }

.bp3-icon-new-drawing::before{
  content:""; }

.bp3-icon-new-grid-item::before{
  content:""; }

.bp3-icon-new-layer::before{
  content:""; }

.bp3-icon-new-layers::before{
  content:""; }

.bp3-icon-new-link::before{
  content:""; }

.bp3-icon-new-object::before{
  content:""; }

.bp3-icon-new-person::before{
  content:""; }

.bp3-icon-new-prescription::before{
  content:""; }

.bp3-icon-new-text-box::before{
  content:""; }

.bp3-icon-ninja::before{
  content:""; }

.bp3-icon-not-equal-to::before{
  content:""; }

.bp3-icon-notifications::before{
  content:""; }

.bp3-icon-notifications-updated::before{
  content:""; }

.bp3-icon-numbered-list::before{
  content:""; }

.bp3-icon-numerical::before{
  content:""; }

.bp3-icon-office::before{
  content:""; }

.bp3-icon-offline::before{
  content:""; }

.bp3-icon-oil-field::before{
  content:""; }

.bp3-icon-one-column::before{
  content:""; }

.bp3-icon-outdated::before{
  content:""; }

.bp3-icon-page-layout::before{
  content:""; }

.bp3-icon-panel-stats::before{
  content:""; }

.bp3-icon-panel-table::before{
  content:""; }

.bp3-icon-paperclip::before{
  content:""; }

.bp3-icon-paragraph::before{
  content:""; }

.bp3-icon-path::before{
  content:""; }

.bp3-icon-path-search::before{
  content:""; }

.bp3-icon-pause::before{
  content:""; }

.bp3-icon-people::before{
  content:""; }

.bp3-icon-percentage::before{
  content:""; }

.bp3-icon-person::before{
  content:""; }

.bp3-icon-phone::before{
  content:"☎"; }

.bp3-icon-pie-chart::before{
  content:""; }

.bp3-icon-pin::before{
  content:""; }

.bp3-icon-pivot::before{
  content:""; }

.bp3-icon-pivot-table::before{
  content:""; }

.bp3-icon-play::before{
  content:""; }

.bp3-icon-plus::before{
  content:"+"; }

.bp3-icon-polygon-filter::before{
  content:""; }

.bp3-icon-power::before{
  content:""; }

.bp3-icon-predictive-analysis::before{
  content:""; }

.bp3-icon-prescription::before{
  content:""; }

.bp3-icon-presentation::before{
  content:""; }

.bp3-icon-print::before{
  content:"⎙"; }

.bp3-icon-projects::before{
  content:""; }

.bp3-icon-properties::before{
  content:""; }

.bp3-icon-property::before{
  content:""; }

.bp3-icon-publish-function::before{
  content:""; }

.bp3-icon-pulse::before{
  content:""; }

.bp3-icon-random::before{
  content:""; }

.bp3-icon-record::before{
  content:""; }

.bp3-icon-redo::before{
  content:""; }

.bp3-icon-refresh::before{
  content:""; }

.bp3-icon-regression-chart::before{
  content:""; }

.bp3-icon-remove::before{
  content:""; }

.bp3-icon-remove-column::before{
  content:""; }

.bp3-icon-remove-column-left::before{
  content:""; }

.bp3-icon-remove-column-right::before{
  content:""; }

.bp3-icon-remove-row-bottom::before{
  content:""; }

.bp3-icon-remove-row-top::before{
  content:""; }

.bp3-icon-repeat::before{
  content:""; }

.bp3-icon-reset::before{
  content:""; }

.bp3-icon-resolve::before{
  content:""; }

.bp3-icon-rig::before{
  content:""; }

.bp3-icon-right-join::before{
  content:""; }

.bp3-icon-ring::before{
  content:""; }

.bp3-icon-rotate-document::before{
  content:""; }

.bp3-icon-rotate-page::before{
  content:""; }

.bp3-icon-satellite::before{
  content:""; }

.bp3-icon-saved::before{
  content:""; }

.bp3-icon-scatter-plot::before{
  content:""; }

.bp3-icon-search::before{
  content:""; }

.bp3-icon-search-around::before{
  content:""; }

.bp3-icon-search-template::before{
  content:""; }

.bp3-icon-search-text::before{
  content:""; }

.bp3-icon-segmented-control::before{
  content:""; }

.bp3-icon-select::before{
  content:""; }

.bp3-icon-selection::before{
  content:"⦿"; }

.bp3-icon-send-to::before{
  content:""; }

.bp3-icon-send-to-graph::before{
  content:""; }

.bp3-icon-send-to-map::before{
  content:""; }

.bp3-icon-series-add::before{
  content:""; }

.bp3-icon-series-configuration::before{
  content:""; }

.bp3-icon-series-derived::before{
  content:""; }

.bp3-icon-series-filtered::before{
  content:""; }

.bp3-icon-series-search::before{
  content:""; }

.bp3-icon-settings::before{
  content:""; }

.bp3-icon-share::before{
  content:""; }

.bp3-icon-shield::before{
  content:""; }

.bp3-icon-shop::before{
  content:""; }

.bp3-icon-shopping-cart::before{
  content:""; }

.bp3-icon-signal-search::before{
  content:""; }

.bp3-icon-sim-card::before{
  content:""; }

.bp3-icon-slash::before{
  content:""; }

.bp3-icon-small-cross::before{
  content:""; }

.bp3-icon-small-minus::before{
  content:""; }

.bp3-icon-small-plus::before{
  content:""; }

.bp3-icon-small-tick::before{
  content:""; }

.bp3-icon-snowflake::before{
  content:""; }

.bp3-icon-social-media::before{
  content:""; }

.bp3-icon-sort::before{
  content:""; }

.bp3-icon-sort-alphabetical::before{
  content:""; }

.bp3-icon-sort-alphabetical-desc::before{
  content:""; }

.bp3-icon-sort-asc::before{
  content:""; }

.bp3-icon-sort-desc::before{
  content:""; }

.bp3-icon-sort-numerical::before{
  content:""; }

.bp3-icon-sort-numerical-desc::before{
  content:""; }

.bp3-icon-split-columns::before{
  content:""; }

.bp3-icon-square::before{
  content:""; }

.bp3-icon-stacked-chart::before{
  content:""; }

.bp3-icon-star::before{
  content:"★"; }

.bp3-icon-star-empty::before{
  content:"☆"; }

.bp3-icon-step-backward::before{
  content:""; }

.bp3-icon-step-chart::before{
  content:""; }

.bp3-icon-step-forward::before{
  content:""; }

.bp3-icon-stop::before{
  content:""; }

.bp3-icon-stopwatch::before{
  content:""; }

.bp3-icon-strikethrough::before{
  content:""; }

.bp3-icon-style::before{
  content:""; }

.bp3-icon-swap-horizontal::before{
  content:""; }

.bp3-icon-swap-vertical::before{
  content:""; }

.bp3-icon-symbol-circle::before{
  content:""; }

.bp3-icon-symbol-cross::before{
  content:""; }

.bp3-icon-symbol-diamond::before{
  content:""; }

.bp3-icon-symbol-square::before{
  content:""; }

.bp3-icon-symbol-triangle-down::before{
  content:""; }

.bp3-icon-symbol-triangle-up::before{
  content:""; }

.bp3-icon-tag::before{
  content:""; }

.bp3-icon-take-action::before{
  content:""; }

.bp3-icon-taxi::before{
  content:""; }

.bp3-icon-text-highlight::before{
  content:""; }

.bp3-icon-th::before{
  content:""; }

.bp3-icon-th-derived::before{
  content:""; }

.bp3-icon-th-disconnect::before{
  content:""; }

.bp3-icon-th-filtered::before{
  content:""; }

.bp3-icon-th-list::before{
  content:""; }

.bp3-icon-thumbs-down::before{
  content:""; }

.bp3-icon-thumbs-up::before{
  content:""; }

.bp3-icon-tick::before{
  content:"✓"; }

.bp3-icon-tick-circle::before{
  content:""; }

.bp3-icon-time::before{
  content:"⏲"; }

.bp3-icon-timeline-area-chart::before{
  content:""; }

.bp3-icon-timeline-bar-chart::before{
  content:""; }

.bp3-icon-timeline-events::before{
  content:""; }

.bp3-icon-timeline-line-chart::before{
  content:""; }

.bp3-icon-tint::before{
  content:""; }

.bp3-icon-torch::before{
  content:""; }

.bp3-icon-tractor::before{
  content:""; }

.bp3-icon-train::before{
  content:""; }

.bp3-icon-translate::before{
  content:""; }

.bp3-icon-trash::before{
  content:""; }

.bp3-icon-tree::before{
  content:""; }

.bp3-icon-trending-down::before{
  content:""; }

.bp3-icon-trending-up::before{
  content:""; }

.bp3-icon-truck::before{
  content:""; }

.bp3-icon-two-columns::before{
  content:""; }

.bp3-icon-unarchive::before{
  content:""; }

.bp3-icon-underline::before{
  content:"⎁"; }

.bp3-icon-undo::before{
  content:"⎌"; }

.bp3-icon-ungroup-objects::before{
  content:""; }

.bp3-icon-unknown-vehicle::before{
  content:""; }

.bp3-icon-unlock::before{
  content:""; }

.bp3-icon-unpin::before{
  content:""; }

.bp3-icon-unresolve::before{
  content:""; }

.bp3-icon-updated::before{
  content:""; }

.bp3-icon-upload::before{
  content:""; }

.bp3-icon-user::before{
  content:""; }

.bp3-icon-variable::before{
  content:""; }

.bp3-icon-vertical-bar-chart-asc::before{
  content:""; }

.bp3-icon-vertical-bar-chart-desc::before{
  content:""; }

.bp3-icon-vertical-distribution::before{
  content:""; }

.bp3-icon-video::before{
  content:""; }

.bp3-icon-volume-down::before{
  content:""; }

.bp3-icon-volume-off::before{
  content:""; }

.bp3-icon-volume-up::before{
  content:""; }

.bp3-icon-walk::before{
  content:""; }

.bp3-icon-warning-sign::before{
  content:""; }

.bp3-icon-waterfall-chart::before{
  content:""; }

.bp3-icon-widget::before{
  content:""; }

.bp3-icon-widget-button::before{
  content:""; }

.bp3-icon-widget-footer::before{
  content:""; }

.bp3-icon-widget-header::before{
  content:""; }

.bp3-icon-wrench::before{
  content:""; }

.bp3-icon-zoom-in::before{
  content:""; }

.bp3-icon-zoom-out::before{
  content:""; }

.bp3-icon-zoom-to-fit::before{
  content:""; }
.bp3-submenu > .bp3-popover-wrapper{
  display:block; }

.bp3-submenu .bp3-popover-target{
  display:block; }

.bp3-submenu.bp3-popover{
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0 5px; }
  .bp3-submenu.bp3-popover > .bp3-popover-content{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-submenu.bp3-popover, .bp3-submenu.bp3-popover.bp3-dark{
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-dark .bp3-submenu.bp3-popover > .bp3-popover-content, .bp3-submenu.bp3-popover.bp3-dark > .bp3-popover-content{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
.bp3-menu{
  margin:0;
  border-radius:3px;
  background:#ffffff;
  min-width:180px;
  padding:5px;
  list-style:none;
  text-align:left;
  color:#182026; }

.bp3-menu-divider{
  display:block;
  margin:5px;
  border-top:1px solid rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-menu-divider{
    border-top-color:rgba(255, 255, 255, 0.15); }

.bp3-menu-item{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  border-radius:2px;
  padding:5px 7px;
  text-decoration:none;
  line-height:20px;
  color:inherit;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-menu-item > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-menu-item > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-menu-item::before,
  .bp3-menu-item > *{
    margin-right:7px; }
  .bp3-menu-item:empty::before,
  .bp3-menu-item > :last-child{
    margin-right:0; }
  .bp3-menu-item > .bp3-fill{
    word-break:break-word; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    background-color:rgba(167, 182, 194, 0.3);
    cursor:pointer;
    text-decoration:none; }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit;
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-dark .bp3-menu-item{
    color:inherit; }
    .bp3-dark .bp3-menu-item:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
      background-color:rgba(138, 155, 168, 0.15);
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-disabled{
      background-color:inherit;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-menu-item.bp3-intent-primary{
    color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-primary::before, .bp3-menu-item.bp3-intent-primary::after,
    .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary:active, .bp3-menu-item.bp3-intent-primary:active::before, .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-success{
    color:#0d8050; }
    .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-success::before, .bp3-menu-item.bp3-intent-success::after,
    .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-menu-item.bp3-intent-success:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success:active, .bp3-menu-item.bp3-intent-success:active::before, .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-warning{
    color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-warning::before, .bp3-menu-item.bp3-intent-warning::after,
    .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning:active, .bp3-menu-item.bp3-intent-warning:active::before, .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-danger{
    color:#c23030; }
    .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-danger::before, .bp3-menu-item.bp3-intent-danger::after,
    .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger:active, .bp3-menu-item.bp3-intent-danger:active::before, .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item::before{
    line-height:1;
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-weight:400;
    font-style:normal;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    margin-right:7px; }
  .bp3-menu-item::before,
  .bp3-menu-item > .bp3-icon{
    margin-top:2px;
    color:#5c7080; }
  .bp3-menu-item .bp3-menu-item-label{
    color:#5c7080; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    color:inherit; }
  .bp3-menu-item.bp3-active, .bp3-menu-item:active{
    background-color:rgba(115, 134, 148, 0.3); }
  .bp3-menu-item.bp3-disabled{
    outline:none !important;
    background-color:inherit !important;
    cursor:not-allowed !important;
    color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-menu-item.bp3-disabled::before,
    .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-large .bp3-menu-item{
    padding:9px 7px;
    line-height:22px;
    font-size:16px; }
    .bp3-large .bp3-menu-item .bp3-icon{
      margin-top:3px; }
    .bp3-large .bp3-menu-item::before{
      line-height:1;
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-weight:400;
      font-style:normal;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      margin-top:1px;
      margin-right:10px; }

button.bp3-menu-item{
  border:none;
  background:none;
  width:100%;
  text-align:left; }
.bp3-menu-header{
  display:block;
  margin:5px;
  border-top:1px solid rgba(16, 22, 26, 0.15);
  cursor:default;
  padding-left:2px; }
  .bp3-dark .bp3-menu-header{
    border-top-color:rgba(255, 255, 255, 0.15); }
  .bp3-menu-header:first-of-type{
    border-top:none; }
  .bp3-menu-header > h6{
    color:#182026;
    font-weight:600;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    margin:0;
    padding:10px 7px 0 1px;
    line-height:17px; }
    .bp3-dark .bp3-menu-header > h6{
      color:#f5f8fa; }
  .bp3-menu-header:first-of-type > h6{
    padding-top:0; }
  .bp3-large .bp3-menu-header > h6{
    padding-top:15px;
    padding-bottom:5px;
    font-size:18px; }
  .bp3-large .bp3-menu-header:first-of-type > h6{
    padding-top:0; }

.bp3-dark .bp3-menu{
  background:#30404d;
  color:#f5f8fa; }

.bp3-dark .bp3-menu-item.bp3-intent-primary{
  color:#48aff0; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-icon{
    color:inherit; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary::before, .bp3-dark .bp3-menu-item.bp3-intent-primary::after,
  .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
    color:#48aff0; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active{
    background-color:#137cbd; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary:active{
    background-color:#106ba3; }
  .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
  .bp3-dark .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
  .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label,
  .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary:active, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
    color:#ffffff; }

.bp3-dark .bp3-menu-item.bp3-intent-success{
  color:#3dcc91; }
  .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-icon{
    color:inherit; }
  .bp3-dark .bp3-menu-item.bp3-intent-success::before, .bp3-dark .bp3-menu-item.bp3-intent-success::after,
  .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
    color:#3dcc91; }
  .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active{
    background-color:#0f9960; }
  .bp3-dark .bp3-menu-item.bp3-intent-success:active{
    background-color:#0d8050; }
  .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
  .bp3-dark .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
  .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label,
  .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success:active, .bp3-dark .bp3-menu-item.bp3-intent-success:active::before, .bp3-dark .bp3-menu-item.bp3-intent-success:active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
    color:#ffffff; }

.bp3-dark .bp3-menu-item.bp3-intent-warning{
  color:#ffb366; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-icon{
    color:inherit; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning::before, .bp3-dark .bp3-menu-item.bp3-intent-warning::after,
  .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
    color:#ffb366; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active{
    background-color:#d9822b; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning:active{
    background-color:#bf7326; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
  .bp3-dark .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
  .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label,
  .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning:active, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
    color:#ffffff; }

.bp3-dark .bp3-menu-item.bp3-intent-danger{
  color:#ff7373; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-icon{
    color:inherit; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger::before, .bp3-dark .bp3-menu-item.bp3-intent-danger::after,
  .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
    color:#ff7373; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active{
    background-color:#db3737; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger:active{
    background-color:#c23030; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
  .bp3-dark .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
  .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label,
  .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger:active, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::after,
  .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
    color:#ffffff; }

.bp3-dark .bp3-menu-item::before,
.bp3-dark .bp3-menu-item > .bp3-icon{
  color:#a7b6c2; }

.bp3-dark .bp3-menu-item .bp3-menu-item-label{
  color:#a7b6c2; }

.bp3-dark .bp3-menu-item.bp3-active, .bp3-dark .bp3-menu-item:active{
  background-color:rgba(138, 155, 168, 0.3); }

.bp3-dark .bp3-menu-item.bp3-disabled{
  color:rgba(167, 182, 194, 0.6) !important; }
  .bp3-dark .bp3-menu-item.bp3-disabled::before,
  .bp3-dark .bp3-menu-item.bp3-disabled > .bp3-icon,
  .bp3-dark .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
    color:rgba(167, 182, 194, 0.6) !important; }

.bp3-dark .bp3-menu-divider,
.bp3-dark .bp3-menu-header{
  border-color:rgba(255, 255, 255, 0.15); }

.bp3-dark .bp3-menu-header > h6{
  color:#f5f8fa; }

.bp3-label .bp3-menu{
  margin-top:5px; }
.bp3-navbar{
  position:relative;
  z-index:10;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  background-color:#ffffff;
  width:100%;
  height:50px;
  padding:0 15px; }
  .bp3-navbar.bp3-dark,
  .bp3-dark .bp3-navbar{
    background-color:#394b59; }
  .bp3-navbar.bp3-dark{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-navbar{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-navbar.bp3-fixed-top{
    position:fixed;
    top:0;
    right:0;
    left:0; }

.bp3-navbar-heading{
  margin-right:15px;
  font-size:16px; }

.bp3-navbar-group{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  height:50px; }
  .bp3-navbar-group.bp3-align-left{
    float:left; }
  .bp3-navbar-group.bp3-align-right{
    float:right; }

.bp3-navbar-divider{
  margin:0 10px;
  border-left:1px solid rgba(16, 22, 26, 0.15);
  height:20px; }
  .bp3-dark .bp3-navbar-divider{
    border-left-color:rgba(255, 255, 255, 0.15); }
.bp3-non-ideal-state{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  width:100%;
  height:100%;
  text-align:center; }
  .bp3-non-ideal-state > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-non-ideal-state > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-non-ideal-state::before,
  .bp3-non-ideal-state > *{
    margin-bottom:20px; }
  .bp3-non-ideal-state:empty::before,
  .bp3-non-ideal-state > :last-child{
    margin-bottom:0; }
  .bp3-non-ideal-state > *{
    max-width:400px; }

.bp3-non-ideal-state-visual{
  color:rgba(92, 112, 128, 0.6);
  font-size:60px; }
  .bp3-dark .bp3-non-ideal-state-visual{
    color:rgba(167, 182, 194, 0.6); }

.bp3-overflow-list{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:nowrap;
      flex-wrap:nowrap;
  min-width:0; }

.bp3-overflow-list-spacer{
  -ms-flex-negative:1;
      flex-shrink:1;
  width:1px; }

body.bp3-overlay-open{
  overflow:hidden; }

.bp3-overlay{
  position:static;
  top:0;
  right:0;
  bottom:0;
  left:0;
  z-index:20; }
  .bp3-overlay:not(.bp3-overlay-open){
    pointer-events:none; }
  .bp3-overlay.bp3-overlay-container{
    position:fixed;
    overflow:hidden; }
    .bp3-overlay.bp3-overlay-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-scroll-container{
    position:fixed;
    overflow:auto; }
    .bp3-overlay.bp3-overlay-scroll-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-inline{
    display:inline;
    overflow:visible; }

.bp3-overlay-content{
  position:fixed;
  z-index:20; }
  .bp3-overlay-inline .bp3-overlay-content,
  .bp3-overlay-scroll-container .bp3-overlay-content{
    position:absolute; }

.bp3-overlay-backdrop{
  position:fixed;
  top:0;
  right:0;
  bottom:0;
  left:0;
  opacity:1;
  z-index:20;
  background-color:rgba(16, 22, 26, 0.7);
  overflow:auto;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-overlay-backdrop.bp3-overlay-enter, .bp3-overlay-backdrop.bp3-overlay-appear{
    opacity:0; }
  .bp3-overlay-backdrop.bp3-overlay-enter-active, .bp3-overlay-backdrop.bp3-overlay-appear-active{
    opacity:1;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-overlay-backdrop.bp3-overlay-exit{
    opacity:1; }
  .bp3-overlay-backdrop.bp3-overlay-exit-active{
    opacity:0;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-overlay-backdrop:focus{
    outline:none; }
  .bp3-overlay-inline .bp3-overlay-backdrop{
    position:absolute; }
.bp3-panel-stack{
  position:relative;
  overflow:hidden; }

.bp3-panel-stack-header{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  z-index:1;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  height:30px; }
  .bp3-dark .bp3-panel-stack-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack-header > span{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1;
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch; }
  .bp3-panel-stack-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack-view{
  position:absolute;
  top:0;
  right:0;
  bottom:0;
  left:0;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  background-color:#ffffff;
  overflow-y:auto; }
  .bp3-dark .bp3-panel-stack-view{
    background-color:#30404d; }

.bp3-panel-stack-push .bp3-panel-stack-enter, .bp3-panel-stack-push .bp3-panel-stack-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack-push .bp3-panel-stack-enter-active, .bp3-panel-stack-push .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease;
  -webkit-transition-delay:0;
          transition-delay:0; }

.bp3-panel-stack-push .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-push .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease;
  -webkit-transition-delay:0;
          transition-delay:0; }

.bp3-panel-stack-pop .bp3-panel-stack-enter, .bp3-panel-stack-pop .bp3-panel-stack-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack-pop .bp3-panel-stack-enter-active, .bp3-panel-stack-pop .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease;
  -webkit-transition-delay:0;
          transition-delay:0; }

.bp3-panel-stack-pop .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-pop .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease;
  -webkit-transition-delay:0;
          transition-delay:0; }
.bp3-popover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1);
  display:inline-block;
  z-index:20;
  border-radius:3px; }
  .bp3-popover .bp3-popover-arrow{
    position:absolute;
    width:30px;
    height:30px; }
    .bp3-popover .bp3-popover-arrow::before{
      margin:5px;
      width:20px;
      height:20px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover{
    margin-top:-17px;
    margin-bottom:17px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
      bottom:-11px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover{
    margin-left:17px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
      left:-11px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover{
    margin-top:17px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
      top:-11px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover{
    margin-right:17px;
    margin-left:-17px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
      right:-11px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-popover > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-popover > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
    top:-0.3934px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
    right:-0.3934px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
    left:-0.3934px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
    bottom:-0.3934px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-popover .bp3-popover-content{
    background:#ffffff;
    color:inherit; }
  .bp3-popover .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-popover .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-popover .bp3-popover-arrow-fill{
    fill:#ffffff; }
  .bp3-popover-enter > .bp3-popover, .bp3-popover-appear > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3); }
  .bp3-popover-enter-active > .bp3-popover, .bp3-popover-appear-active > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-popover-exit > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-popover .bp3-popover-content{
    position:relative;
    border-radius:3px; }
  .bp3-popover.bp3-popover-content-sizing .bp3-popover-content{
    max-width:350px;
    padding:20px; }
  .bp3-popover-target + .bp3-overlay .bp3-popover.bp3-popover-content-sizing{
    width:350px; }
  .bp3-popover.bp3-minimal{
    margin:0 !important; }
    .bp3-popover.bp3-minimal .bp3-popover-arrow{
      display:none; }
    .bp3-popover.bp3-minimal.bp3-popover{
      -webkit-transform:scale(1);
              transform:scale(1); }
      .bp3-popover-enter > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-enter-active > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
        -webkit-transition-delay:0;
                transition-delay:0; }
      .bp3-popover-exit > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-exit-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
        -webkit-transition-delay:0;
                transition-delay:0; }
  .bp3-popover.bp3-dark,
  .bp3-dark .bp3-popover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-popover .bp3-popover-content{
      background:#30404d;
      color:inherit; }
    .bp3-popover.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-popover .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-popover .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-popover.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-popover .bp3-popover-arrow-fill{
      fill:#30404d; }

.bp3-popover-arrow::before{
  display:block;
  position:absolute;
  -webkit-transform:rotate(45deg);
          transform:rotate(45deg);
  border-radius:2px;
  content:""; }

.bp3-tether-pinned .bp3-popover-arrow{
  display:none; }

.bp3-popover-backdrop{
  background:rgba(255, 255, 255, 0); }

.bp3-transition-container{
  opacity:1;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  z-index:20; }
  .bp3-transition-container.bp3-popover-enter, .bp3-transition-container.bp3-popover-appear{
    opacity:0; }
  .bp3-transition-container.bp3-popover-enter-active, .bp3-transition-container.bp3-popover-appear-active{
    opacity:1;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-transition-container.bp3-popover-exit{
    opacity:1; }
  .bp3-transition-container.bp3-popover-exit-active{
    opacity:0;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-transition-container:focus{
    outline:none; }
  .bp3-transition-container.bp3-popover-leave .bp3-popover-content{
    pointer-events:none; }
  .bp3-transition-container[data-x-out-of-boundaries]{
    display:none; }

span.bp3-popover-target{
  display:inline-block; }

.bp3-popover-wrapper.bp3-fill{
  width:100%; }

.bp3-portal{
  position:absolute;
  top:0;
  right:0;
  left:0; }
@-webkit-keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }
@keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }

.bp3-progress-bar{
  display:block;
  position:relative;
  border-radius:40px;
  background:rgba(92, 112, 128, 0.2);
  width:100%;
  height:8px;
  overflow:hidden; }
  .bp3-progress-bar .bp3-progress-meter{
    position:absolute;
    border-radius:40px;
    background:linear-gradient(-45deg, rgba(255, 255, 255, 0.2) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.2) 50%, rgba(255, 255, 255, 0.2) 75%, transparent 75%);
    background-color:rgba(92, 112, 128, 0.8);
    background-size:30px 30px;
    width:100%;
    height:100%;
    -webkit-transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-progress-bar:not(.bp3-no-animation):not(.bp3-no-stripes) .bp3-progress-meter{
    animation:linear-progress-bar-stripes 300ms linear infinite reverse; }
  .bp3-progress-bar.bp3-no-stripes .bp3-progress-meter{
    background-image:none; }

.bp3-dark .bp3-progress-bar{
  background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-progress-bar .bp3-progress-meter{
    background-color:#8a9ba8; }

.bp3-progress-bar.bp3-intent-primary .bp3-progress-meter{
  background-color:#137cbd; }

.bp3-progress-bar.bp3-intent-success .bp3-progress-meter{
  background-color:#0f9960; }

.bp3-progress-bar.bp3-intent-warning .bp3-progress-meter{
  background-color:#d9822b; }

.bp3-progress-bar.bp3-intent-danger .bp3-progress-meter{
  background-color:#db3737; }
@-webkit-keyframes skeleton-glow{
  from{
    border-color:rgba(206, 217, 224, 0.2);
    background:rgba(206, 217, 224, 0.2); }
  to{
    border-color:rgba(92, 112, 128, 0.2);
    background:rgba(92, 112, 128, 0.2); } }
@keyframes skeleton-glow{
  from{
    border-color:rgba(206, 217, 224, 0.2);
    background:rgba(206, 217, 224, 0.2); }
  to{
    border-color:rgba(92, 112, 128, 0.2);
    background:rgba(92, 112, 128, 0.2); } }
.bp3-skeleton{
  border-color:rgba(206, 217, 224, 0.2) !important;
  border-radius:2px;
  -webkit-box-shadow:none !important;
          box-shadow:none !important;
  background:rgba(206, 217, 224, 0.2);
  background-clip:padding-box !important;
  cursor:default;
  color:transparent !important;
  -webkit-animation:1000ms linear infinite alternate skeleton-glow;
          animation:1000ms linear infinite alternate skeleton-glow;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-skeleton::before, .bp3-skeleton::after,
  .bp3-skeleton *{
    visibility:hidden !important; }
.bp3-slider{
  width:100%;
  min-width:150px;
  height:40px;
  position:relative;
  outline:none;
  cursor:default;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-slider:hover{
    cursor:pointer; }
  .bp3-slider:active{
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-slider.bp3-disabled{
    opacity:0.5;
    cursor:not-allowed; }
  .bp3-slider.bp3-slider-unlabeled{
    height:16px; }

.bp3-slider-track,
.bp3-slider-progress{
  top:5px;
  right:0;
  left:0;
  height:6px;
  position:absolute; }

.bp3-slider-track{
  border-radius:3px;
  overflow:hidden; }

.bp3-slider-progress{
  background:rgba(92, 112, 128, 0.2); }
  .bp3-dark .bp3-slider-progress{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-slider-progress.bp3-intent-primary{
    background-color:#137cbd; }
  .bp3-slider-progress.bp3-intent-success{
    background-color:#0f9960; }
  .bp3-slider-progress.bp3-intent-warning{
    background-color:#d9822b; }
  .bp3-slider-progress.bp3-intent-danger{
    background-color:#db3737; }

.bp3-slider-handle{
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  color:#182026;
  position:absolute;
  top:0;
  left:0;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
  cursor:pointer;
  width:16px;
  height:16px; }
  .bp3-slider-handle:hover{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#ebf1f5; }
  .bp3-slider-handle:active, .bp3-slider-handle.bp3-active{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#d8e1e8;
    background-image:none; }
  .bp3-slider-handle:disabled, .bp3-slider-handle.bp3-disabled{
    outline:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
    .bp3-slider-handle:disabled.bp3-active, .bp3-slider-handle:disabled.bp3-active:hover, .bp3-slider-handle.bp3-disabled.bp3-active, .bp3-slider-handle.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }
  .bp3-slider-handle:focus{
    z-index:1; }
  .bp3-slider-handle:hover{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    background-clip:padding-box;
    background-color:#ebf1f5;
    z-index:2;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
    cursor:-webkit-grab;
    cursor:grab; }
  .bp3-slider-handle.bp3-active{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-disabled .bp3-slider-handle{
    -webkit-box-shadow:none;
            box-shadow:none;
    background:#bfccd6;
    pointer-events:none; }
  .bp3-dark .bp3-slider-handle{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover, .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      background-color:#30404d; }
    .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      background-color:#202b33;
      background-image:none; }
    .bp3-dark .bp3-slider-handle:disabled, .bp3-dark .bp3-slider-handle.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none;
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-slider-handle:disabled.bp3-active, .bp3-dark .bp3-slider-handle.bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-slider-handle .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-slider-handle, .bp3-dark .bp3-slider-handle:hover{
      background-color:#394b59; }
    .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#293742; }
  .bp3-dark .bp3-disabled .bp3-slider-handle{
    border-color:#5c7080;
    -webkit-box-shadow:none;
            box-shadow:none;
    background:#5c7080; }
  .bp3-slider-handle .bp3-slider-label{
    margin-left:8px;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    background:#394b59;
    color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle .bp3-slider-label{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
      background:#e1e8ed;
      color:#394b59; }
    .bp3-disabled .bp3-slider-handle .bp3-slider-label{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-slider-handle.bp3-start, .bp3-slider-handle.bp3-end{
    width:8px; }
  .bp3-slider-handle.bp3-start{
    border-top-right-radius:0;
    border-bottom-right-radius:0; }
  .bp3-slider-handle.bp3-end{
    margin-left:8px;
    border-top-left-radius:0;
    border-bottom-left-radius:0; }
    .bp3-slider-handle.bp3-end .bp3-slider-label{
      margin-left:0; }

.bp3-slider-label{
  -webkit-transform:translate(-50%, 20px);
          transform:translate(-50%, 20px);
  display:inline-block;
  position:absolute;
  padding:2px 5px;
  vertical-align:top;
  line-height:1;
  font-size:12px; }

.bp3-slider.bp3-vertical{
  width:40px;
  min-width:40px;
  height:150px; }
  .bp3-slider.bp3-vertical .bp3-slider-track,
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    top:0;
    bottom:0;
    left:5px;
    width:6px;
    height:auto; }
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    top:auto; }
  .bp3-slider.bp3-vertical .bp3-slider-label{
    -webkit-transform:translate(20px, 50%);
            transform:translate(20px, 50%); }
  .bp3-slider.bp3-vertical .bp3-slider-handle{
    top:auto; }
    .bp3-slider.bp3-vertical .bp3-slider-handle .bp3-slider-label{
      margin-top:-8px;
      margin-left:0; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end, .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      margin-left:0;
      width:16px;
      height:8px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      border-top-left-radius:0;
      border-bottom-right-radius:3px; }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start .bp3-slider-label{
        -webkit-transform:translate(20px);
                transform:translate(20px); }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end{
      margin-bottom:8px;
      border-top-left-radius:3px;
      border-bottom-left-radius:0;
      border-bottom-right-radius:0; }

@-webkit-keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

@keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

.bp3-spinner{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  overflow:visible;
  vertical-align:middle; }
  .bp3-spinner svg{
    display:block; }
  .bp3-spinner path{
    fill-opacity:0; }
  .bp3-spinner .bp3-spinner-head{
    -webkit-transform-origin:center;
            transform-origin:center;
    -webkit-transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    stroke:rgba(92, 112, 128, 0.8);
    stroke-linecap:round; }
  .bp3-spinner .bp3-spinner-track{
    stroke:rgba(92, 112, 128, 0.2); }

.bp3-spinner-animation{
  -webkit-animation:pt-spinner-animation 500ms linear infinite;
          animation:pt-spinner-animation 500ms linear infinite; }
  .bp3-no-spin > .bp3-spinner-animation{
    -webkit-animation:none;
            animation:none; }

.bp3-dark .bp3-spinner .bp3-spinner-head{
  stroke:#8a9ba8; }

.bp3-dark .bp3-spinner .bp3-spinner-track{
  stroke:rgba(16, 22, 26, 0.5); }

.bp3-spinner.bp3-intent-primary .bp3-spinner-head{
  stroke:#137cbd; }

.bp3-spinner.bp3-intent-success .bp3-spinner-head{
  stroke:#0f9960; }

.bp3-spinner.bp3-intent-warning .bp3-spinner-head{
  stroke:#d9822b; }

.bp3-spinner.bp3-intent-danger .bp3-spinner-head{
  stroke:#db3737; }
.bp3-tabs.bp3-vertical{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-tabs.bp3-vertical > .bp3-tab-list{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab{
      border-radius:3px;
      width:100%;
      padding:0 10px; }
      .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab[aria-selected="true"]{
        -webkit-box-shadow:none;
                box-shadow:none;
        background-color:rgba(19, 124, 189, 0.2); }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab-indicator-wrapper .bp3-tab-indicator{
      top:0;
      right:0;
      bottom:0;
      left:0;
      border-radius:3px;
      background-color:rgba(19, 124, 189, 0.2);
      height:auto; }
  .bp3-tabs.bp3-vertical > .bp3-tab-panel{
    margin-top:0;
    padding-left:20px; }

.bp3-tab-list{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  -webkit-box-align:end;
      -ms-flex-align:end;
          align-items:flex-end;
  position:relative;
  margin:0;
  border:none;
  padding:0;
  list-style:none; }
  .bp3-tab-list > *:not(:last-child){
    margin-right:20px; }

.bp3-tab{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  position:relative;
  cursor:pointer;
  max-width:100%;
  vertical-align:top;
  line-height:30px;
  color:#182026;
  font-size:14px; }
  .bp3-tab a{
    display:block;
    text-decoration:none;
    color:inherit; }
  .bp3-tab-indicator-wrapper ~ .bp3-tab{
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    background-color:transparent !important; }
  .bp3-tab[aria-disabled="true"]{
    cursor:not-allowed;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-tab[aria-selected="true"]{
    border-radius:0;
    -webkit-box-shadow:inset 0 -3px 0 #106ba3;
            box-shadow:inset 0 -3px 0 #106ba3; }
  .bp3-tab[aria-selected="true"], .bp3-tab:not([aria-disabled="true"]):hover{
    color:#106ba3; }
  .bp3-tab:focus{
    -moz-outline-radius:0; }
  .bp3-large > .bp3-tab{
    line-height:40px;
    font-size:16px; }

.bp3-tab-panel{
  margin-top:20px; }
  .bp3-tab-panel[aria-hidden="true"]{
    display:none; }

.bp3-tab-indicator-wrapper{
  position:absolute;
  top:0;
  left:0;
  -webkit-transform:translateX(0), translateY(0);
          transform:translateX(0), translateY(0);
  -webkit-transition:height, width, -webkit-transform;
  transition:height, width, -webkit-transform;
  transition:height, transform, width;
  transition:height, transform, width, -webkit-transform;
  -webkit-transition-duration:200ms;
          transition-duration:200ms;
  -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
          transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
  pointer-events:none; }
  .bp3-tab-indicator-wrapper .bp3-tab-indicator{
    position:absolute;
    right:0;
    bottom:0;
    left:0;
    background-color:#106ba3;
    height:3px; }
  .bp3-tab-indicator-wrapper.bp3-no-animation{
    -webkit-transition:none;
    transition:none; }

.bp3-dark .bp3-tab{
  color:#f5f8fa; }
  .bp3-dark .bp3-tab[aria-disabled="true"]{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tab[aria-selected="true"]{
    -webkit-box-shadow:inset 0 -3px 0 #48aff0;
            box-shadow:inset 0 -3px 0 #48aff0; }
  .bp3-dark .bp3-tab[aria-selected="true"], .bp3-dark .bp3-tab:not([aria-disabled="true"]):hover{
    color:#48aff0; }

.bp3-dark .bp3-tab-indicator{
  background-color:#48aff0; }

.bp3-flex-expander{
  -webkit-box-flex:1;
      -ms-flex:1 1;
          flex:1 1; }
.bp3-tag{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  position:relative;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:none;
          box-shadow:none;
  background-color:#5c7080;
  min-width:20px;
  max-width:100%;
  min-height:20px;
  padding:2px 6px;
  line-height:16px;
  color:#f5f8fa;
  font-size:12px; }
  .bp3-tag.bp3-interactive{
    cursor:pointer; }
    .bp3-tag.bp3-interactive:hover{
      background-color:rgba(92, 112, 128, 0.85); }
    .bp3-tag.bp3-interactive.bp3-active, .bp3-tag.bp3-interactive:active{
      background-color:rgba(92, 112, 128, 0.7); }
  .bp3-tag > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag::before,
  .bp3-tag > *{
    margin-right:4px; }
  .bp3-tag:empty::before,
  .bp3-tag > :last-child{
    margin-right:0; }
  .bp3-tag:focus{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:0;
    -moz-outline-radius:6px; }
  .bp3-tag.bp3-round{
    border-radius:30px;
    padding-right:8px;
    padding-left:8px; }
  .bp3-dark .bp3-tag{
    background-color:#bfccd6;
    color:#182026; }
    .bp3-dark .bp3-tag.bp3-interactive{
      cursor:pointer; }
      .bp3-dark .bp3-tag.bp3-interactive:hover{
        background-color:rgba(191, 204, 214, 0.85); }
      .bp3-dark .bp3-tag.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-interactive:active{
        background-color:rgba(191, 204, 214, 0.7); }
    .bp3-dark .bp3-tag > .bp3-icon, .bp3-dark .bp3-tag .bp3-icon-standard, .bp3-dark .bp3-tag .bp3-icon-large{
      fill:currentColor; }
  .bp3-tag > .bp3-icon, .bp3-tag .bp3-icon-standard, .bp3-tag .bp3-icon-large{
    fill:#ffffff; }
  .bp3-tag.bp3-large,
  .bp3-large .bp3-tag{
    min-width:30px;
    min-height:30px;
    padding:0 10px;
    line-height:20px;
    font-size:14px; }
    .bp3-tag.bp3-large::before,
    .bp3-tag.bp3-large > *,
    .bp3-large .bp3-tag::before,
    .bp3-large .bp3-tag > *{
      margin-right:7px; }
    .bp3-tag.bp3-large:empty::before,
    .bp3-tag.bp3-large > :last-child,
    .bp3-large .bp3-tag:empty::before,
    .bp3-large .bp3-tag > :last-child{
      margin-right:0; }
    .bp3-tag.bp3-large.bp3-round,
    .bp3-large .bp3-tag.bp3-round{
      padding-right:12px;
      padding-left:12px; }
  .bp3-tag.bp3-intent-primary{
    background:#137cbd;
    color:#ffffff; }
    .bp3-tag.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.85); }
      .bp3-tag.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.7); }
  .bp3-tag.bp3-intent-success{
    background:#0f9960;
    color:#ffffff; }
    .bp3-tag.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.85); }
      .bp3-tag.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.7); }
  .bp3-tag.bp3-intent-warning{
    background:#d9822b;
    color:#ffffff; }
    .bp3-tag.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.85); }
      .bp3-tag.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.7); }
  .bp3-tag.bp3-intent-danger{
    background:#db3737;
    color:#ffffff; }
    .bp3-tag.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.85); }
      .bp3-tag.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.7); }
  .bp3-tag.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-tag.bp3-minimal > .bp3-icon, .bp3-tag.bp3-minimal .bp3-icon-standard, .bp3-tag.bp3-minimal .bp3-icon-large{
    fill:#5c7080; }
  .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
    background-color:rgba(138, 155, 168, 0.2);
    color:#182026; }
    .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
        background-color:rgba(92, 112, 128, 0.3); }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
        background-color:rgba(92, 112, 128, 0.4); }
    .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
      color:#f5f8fa; }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
          background-color:rgba(191, 204, 214, 0.3); }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
          background-color:rgba(191, 204, 214, 0.4); }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) > .bp3-icon, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-large{
        fill:#a7b6c2; }
  .bp3-tag.bp3-minimal.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15);
    color:#106ba3; }
    .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-primary > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-large{
      fill:#137cbd; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25);
      color:#48aff0; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
          background-color:rgba(19, 124, 189, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
          background-color:rgba(19, 124, 189, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15);
    color:#0d8050; }
    .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-success > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-large{
      fill:#0f9960; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25);
      color:#3dcc91; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
          background-color:rgba(15, 153, 96, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
          background-color:rgba(15, 153, 96, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15);
    color:#bf7326; }
    .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-warning > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-large{
      fill:#d9822b; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25);
      color:#ffb366; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
          background-color:rgba(217, 130, 43, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
          background-color:rgba(217, 130, 43, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15);
    color:#c23030; }
    .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-danger > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-large{
      fill:#db3737; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25);
      color:#ff7373; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
          background-color:rgba(219, 55, 55, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
          background-color:rgba(219, 55, 55, 0.45); }

.bp3-tag-remove{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  opacity:0.5;
  margin-top:-2px;
  margin-right:-6px !important;
  margin-bottom:-2px;
  border:none;
  background:none;
  cursor:pointer;
  padding:2px;
  padding-left:0;
  color:inherit; }
  .bp3-tag-remove:hover{
    opacity:0.8;
    background:none;
    text-decoration:none; }
  .bp3-tag-remove:active{
    opacity:1; }
  .bp3-tag-remove:empty::before{
    line-height:1;
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-weight:400;
    font-style:normal;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    content:""; }
  .bp3-large .bp3-tag-remove{
    margin-right:-10px !important;
    padding:5px;
    padding-left:0; }
    .bp3-large .bp3-tag-remove:empty::before{
      line-height:1;
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-weight:400;
      font-style:normal; }
.bp3-tag-input{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  cursor:text;
  height:auto;
  min-height:30px;
  padding-right:0;
  padding-left:5px;
  line-height:inherit; }
  .bp3-tag-input > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag-input > .bp3-tag-input-values{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag-input .bp3-tag-input-icon{
    margin-top:7px;
    margin-right:7px;
    margin-left:2px;
    color:#5c7080; }
  .bp3-tag-input .bp3-tag-input-values{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -ms-flex-wrap:wrap;
        flex-wrap:wrap;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    -ms-flex-item-align:stretch;
        align-self:stretch;
    margin-top:5px;
    margin-right:7px;
    min-width:0; }
    .bp3-tag-input .bp3-tag-input-values > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-tag-input .bp3-tag-input-values > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-tag-input .bp3-tag-input-values::before,
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-right:5px; }
    .bp3-tag-input .bp3-tag-input-values:empty::before,
    .bp3-tag-input .bp3-tag-input-values > :last-child{
      margin-right:0; }
    .bp3-tag-input .bp3-tag-input-values:first-child .bp3-input-ghost:first-child{
      padding-left:5px; }
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-bottom:5px; }
  .bp3-tag-input .bp3-tag{
    overflow-wrap:break-word; }
    .bp3-tag-input .bp3-tag.bp3-active{
      outline:rgba(19, 124, 189, 0.6) auto 2px;
      outline-offset:0;
      -moz-outline-radius:6px; }
  .bp3-tag-input .bp3-input-ghost{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:80px;
    line-height:20px; }
    .bp3-tag-input .bp3-input-ghost:disabled, .bp3-tag-input .bp3-input-ghost.bp3-disabled{
      cursor:not-allowed; }
  .bp3-tag-input .bp3-button,
  .bp3-tag-input .bp3-spinner{
    margin:3px;
    margin-left:0; }
  .bp3-tag-input .bp3-button{
    min-width:24px;
    min-height:24px;
    padding:0 7px; }
  .bp3-tag-input.bp3-large{
    height:auto;
    min-height:40px; }
    .bp3-tag-input.bp3-large::before,
    .bp3-tag-input.bp3-large > *{
      margin-right:10px; }
    .bp3-tag-input.bp3-large:empty::before,
    .bp3-tag-input.bp3-large > :last-child{
      margin-right:0; }
    .bp3-tag-input.bp3-large .bp3-tag-input-icon{
      margin-top:10px;
      margin-left:5px; }
    .bp3-tag-input.bp3-large .bp3-input-ghost{
      line-height:30px; }
    .bp3-tag-input.bp3-large .bp3-button{
      min-width:30px;
      min-height:30px;
      padding:5px 10px;
      margin:5px;
      margin-left:0; }
    .bp3-tag-input.bp3-large .bp3-spinner{
      margin:8px;
      margin-left:0; }
  .bp3-tag-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
    background-color:#ffffff; }
    .bp3-tag-input.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-tag-input .bp3-tag-input-icon, .bp3-tag-input.bp3-dark .bp3-tag-input-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-tag-input .bp3-input-ghost, .bp3-tag-input.bp3-dark .bp3-input-ghost{
    color:#f5f8fa; }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-webkit-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-moz-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost:-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::placeholder{
      color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tag-input.bp3-active, .bp3-tag-input.bp3-dark.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    background-color:rgba(16, 22, 26, 0.3); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-primary, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-success, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-warning, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-danger, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-input-ghost{
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  background:none;
  padding:0; }
  .bp3-input-ghost::-webkit-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost::-moz-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost:-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost::-ms-input-placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost::placeholder{
    opacity:1;
    color:rgba(92, 112, 128, 0.6); }
  .bp3-input-ghost:focus{
    outline:none !important; }
.bp3-toast{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  position:relative !important;
  margin:20px 0 0;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  background-color:#ffffff;
  min-width:300px;
  max-width:500px;
  pointer-events:all; }
  .bp3-toast.bp3-toast-enter, .bp3-toast.bp3-toast-appear{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active, .bp3-toast.bp3-toast-appear-active{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-toast.bp3-toast-enter ~ .bp3-toast, .bp3-toast.bp3-toast-appear ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active ~ .bp3-toast, .bp3-toast.bp3-toast-appear-active ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-toast.bp3-toast-exit{
    opacity:1;
    -webkit-filter:blur(0);
            filter:blur(0); }
  .bp3-toast.bp3-toast-exit-active{
    opacity:0;
    -webkit-filter:blur(10px);
            filter:blur(10px);
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:opacity, filter;
    transition-property:opacity, filter, -webkit-filter;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-toast.bp3-toast-exit ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0); }
  .bp3-toast.bp3-toast-exit-active ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:50ms;
            transition-delay:50ms; }
  .bp3-toast .bp3-button-group{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    padding:5px;
    padding-left:0; }
  .bp3-toast > .bp3-icon{
    margin:12px;
    margin-right:0;
    color:#5c7080; }
  .bp3-toast.bp3-dark,
  .bp3-dark .bp3-toast{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
    background-color:#394b59; }
    .bp3-toast.bp3-dark > .bp3-icon,
    .bp3-dark .bp3-toast > .bp3-icon{
      color:#a7b6c2; }
  .bp3-toast[class*="bp3-intent-"] a{
    color:rgba(255, 255, 255, 0.7); }
    .bp3-toast[class*="bp3-intent-"] a:hover{
      color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] > .bp3-icon{
    color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button, .bp3-toast[class*="bp3-intent-"] .bp3-button::before,
  .bp3-toast[class*="bp3-intent-"] .bp3-button .bp3-icon, .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    color:rgba(255, 255, 255, 0.7) !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:focus{
    outline-color:rgba(255, 255, 255, 0.5); }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:hover{
    background-color:rgba(255, 255, 255, 0.15) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    background-color:rgba(255, 255, 255, 0.3) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button::after{
    background:rgba(255, 255, 255, 0.3) !important; }
  .bp3-toast.bp3-intent-primary{
    background-color:#137cbd;
    color:#ffffff; }
  .bp3-toast.bp3-intent-success{
    background-color:#0f9960;
    color:#ffffff; }
  .bp3-toast.bp3-intent-warning{
    background-color:#d9822b;
    color:#ffffff; }
  .bp3-toast.bp3-intent-danger{
    background-color:#db3737;
    color:#ffffff; }

.bp3-toast-message{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  padding:11px;
  word-break:break-word; }

.bp3-toast-container{
  display:-webkit-box !important;
  display:-ms-flexbox !important;
  display:flex !important;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  position:fixed;
  right:0;
  left:0;
  z-index:40;
  overflow:hidden;
  padding:0 20px 20px;
  pointer-events:none; }
  .bp3-toast-container.bp3-toast-container-top{
    top:0;
    bottom:auto; }
  .bp3-toast-container.bp3-toast-container-bottom{
    -webkit-box-orient:vertical;
    -webkit-box-direction:reverse;
        -ms-flex-direction:column-reverse;
            flex-direction:column-reverse;
    top:auto;
    bottom:0; }
  .bp3-toast-container.bp3-toast-container-left{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
  .bp3-toast-container.bp3-toast-container-right{
    -webkit-box-align:end;
        -ms-flex-align:end;
            align-items:flex-end; }

.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active) ~ .bp3-toast, .bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active) ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-leave-active ~ .bp3-toast{
  -webkit-transform:translateY(60px);
          transform:translateY(60px); }
.bp3-tooltip{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1); }
  .bp3-tooltip .bp3-popover-arrow{
    position:absolute;
    width:22px;
    height:22px; }
    .bp3-tooltip .bp3-popover-arrow::before{
      margin:4px;
      width:14px;
      height:14px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip{
    margin-top:-11px;
    margin-bottom:11px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
      bottom:-8px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip{
    margin-left:11px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
      left:-8px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip{
    margin-top:11px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
      top:-8px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip{
    margin-right:11px;
    margin-left:-11px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
      right:-8px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-tooltip > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-tooltip > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
    top:-0.22183px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
    right:-0.22183px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
    left:-0.22183px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
    bottom:-0.22183px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-tooltip .bp3-popover-content{
    background:#394b59;
    color:#f5f8fa; }
  .bp3-tooltip .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-tooltip .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-tooltip .bp3-popover-arrow-fill{
    fill:#394b59; }
  .bp3-popover-enter > .bp3-tooltip, .bp3-popover-appear > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8); }
  .bp3-popover-enter-active > .bp3-tooltip, .bp3-popover-appear-active > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-popover-exit > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8);
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-tooltip .bp3-popover-content{
    padding:10px 12px; }
  .bp3-tooltip.bp3-dark,
  .bp3-dark .bp3-tooltip{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-tooltip .bp3-popover-content{
      background:#e1e8ed;
      color:#394b59; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-fill{
      fill:#e1e8ed; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-content{
    background:#137cbd;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-arrow-fill{
    fill:#137cbd; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-content{
    background:#0f9960;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-arrow-fill{
    fill:#0f9960; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-content{
    background:#d9822b;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-arrow-fill{
    fill:#d9822b; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-content{
    background:#db3737;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-arrow-fill{
    fill:#db3737; }

.bp3-tooltip-indicator{
  border-bottom:dotted 1px;
  cursor:help; }
.bp3-tree .bp3-icon, .bp3-tree .bp3-icon-standard, .bp3-tree .bp3-icon-large{
  color:#5c7080; }
  .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-tree .bp3-icon.bp3-intent-success, .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-tree-node-list{
  margin:0;
  padding-left:0;
  list-style:none; }

.bp3-tree-root{
  position:relative;
  background-color:transparent;
  cursor:default;
  padding-left:0; }

.bp3-tree-node-content-0{
  padding-left:0px; }

.bp3-tree-node-content-1{
  padding-left:23px; }

.bp3-tree-node-content-2{
  padding-left:46px; }

.bp3-tree-node-content-3{
  padding-left:69px; }

.bp3-tree-node-content-4{
  padding-left:92px; }

.bp3-tree-node-content-5{
  padding-left:115px; }

.bp3-tree-node-content-6{
  padding-left:138px; }

.bp3-tree-node-content-7{
  padding-left:161px; }

.bp3-tree-node-content-8{
  padding-left:184px; }

.bp3-tree-node-content-9{
  padding-left:207px; }

.bp3-tree-node-content-10{
  padding-left:230px; }

.bp3-tree-node-content-11{
  padding-left:253px; }

.bp3-tree-node-content-12{
  padding-left:276px; }

.bp3-tree-node-content-13{
  padding-left:299px; }

.bp3-tree-node-content-14{
  padding-left:322px; }

.bp3-tree-node-content-15{
  padding-left:345px; }

.bp3-tree-node-content-16{
  padding-left:368px; }

.bp3-tree-node-content-17{
  padding-left:391px; }

.bp3-tree-node-content-18{
  padding-left:414px; }

.bp3-tree-node-content-19{
  padding-left:437px; }

.bp3-tree-node-content-20{
  padding-left:460px; }

.bp3-tree-node-content{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  width:100%;
  height:30px;
  padding-right:5px; }
  .bp3-tree-node-content:hover{
    background-color:rgba(191, 204, 214, 0.4); }

.bp3-tree-node-caret,
.bp3-tree-node-caret-none{
  min-width:30px; }

.bp3-tree-node-caret{
  color:#5c7080;
  -webkit-transform:rotate(0deg);
          transform:rotate(0deg);
  cursor:pointer;
  padding:7px;
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tree-node-caret:hover{
    color:#182026; }
  .bp3-dark .bp3-tree-node-caret{
    color:#a7b6c2; }
    .bp3-dark .bp3-tree-node-caret:hover{
      color:#f5f8fa; }
  .bp3-tree-node-caret.bp3-tree-node-caret-open{
    -webkit-transform:rotate(90deg);
            transform:rotate(90deg); }
  .bp3-tree-node-caret.bp3-icon-standard::before{
    content:""; }

.bp3-tree-node-icon{
  position:relative;
  margin-right:7px; }

.bp3-tree-node-label{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-label span{
    display:inline; }

.bp3-tree-node-secondary-label{
  padding:0 5px;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-secondary-label .bp3-popover-wrapper,
  .bp3-tree-node-secondary-label .bp3-popover-target{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-content{
  background-color:inherit;
  cursor:not-allowed;
  color:rgba(92, 112, 128, 0.6); }

.bp3-tree-node.bp3-disabled .bp3-tree-node-caret,
.bp3-tree-node.bp3-disabled .bp3-tree-node-icon{
  cursor:not-allowed;
  color:rgba(92, 112, 128, 0.6); }

.bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content,
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-standard, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-large{
    color:#ffffff; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret::before{
    color:rgba(255, 255, 255, 0.7); }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret:hover::before{
    color:#ffffff; }

.bp3-dark .bp3-tree-node-content:hover{
  background-color:rgba(92, 112, 128, 0.3); }

.bp3-dark .bp3-tree .bp3-icon, .bp3-dark .bp3-tree .bp3-icon-standard, .bp3-dark .bp3-tree .bp3-icon-large{
  color:#a7b6c2; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-dark .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
/*!

Copyright 2017-present Palantir Technologies, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.

*/
.bp3-omnibar{
  -webkit-filter:blur(0);
          filter:blur(0);
  opacity:1;
  top:20vh;
  left:calc(50% - 250px);
  z-index:21;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  background-color:#ffffff;
  width:500px; }
  .bp3-omnibar.bp3-overlay-enter, .bp3-omnibar.bp3-overlay-appear{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2; }
  .bp3-omnibar.bp3-overlay-enter-active, .bp3-omnibar.bp3-overlay-appear-active{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-omnibar.bp3-overlay-exit{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1; }
  .bp3-omnibar.bp3-overlay-exit-active{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
    -webkit-transition-delay:0;
            transition-delay:0; }
  .bp3-omnibar .bp3-input{
    border-radius:0;
    background-color:transparent; }
    .bp3-omnibar .bp3-input, .bp3-omnibar .bp3-input:focus{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-omnibar .bp3-menu{
    border-radius:0;
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
    background-color:transparent;
    max-height:calc(60vh - 40px);
    overflow:auto; }
    .bp3-omnibar .bp3-menu:empty{
      display:none; }
  .bp3-dark .bp3-omnibar, .bp3-omnibar.bp3-dark{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    background-color:#30404d; }

.bp3-omnibar-overlay .bp3-overlay-backdrop{
  background-color:rgba(16, 22, 26, 0.2); }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-width:400px;
  max-height:300px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }

.bp3-multi-select{
  min-width:150px; }

.bp3-multi-select-popover .bp3-menu{
  max-width:400px;
  max-height:300px;
  overflow:auto; }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-width:400px;
  max-height:300px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDhoLTIuODFjLS40NS0uNzgtMS4wNy0xLjQ1LTEuODItMS45NkwxNyA0LjQxIDE1LjU5IDNsLTIuMTcgMi4xN0MxMi45NiA1LjA2IDEyLjQ5IDUgMTIgNWMtLjQ5IDAtLjk2LjA2LTEuNDEuMTdMOC40MSAzIDcgNC40MWwxLjYyIDEuNjNDNy44OCA2LjU1IDcuMjYgNy4yMiA2LjgxIDhINHYyaDIuMDljLS4wNS4zMy0uMDkuNjYtLjA5IDF2MUg0djJoMnYxYzAgLjM0LjA0LjY3LjA5IDFINHYyaDIuODFjMS4wNCAxLjc5IDIuOTcgMyA1LjE5IDNzNC4xNS0xLjIxIDUuMTktM0gyMHYtMmgtMi4wOWMuMDUtLjMzLjA5LS42Ni4wOS0xdi0xaDJ2LTJoLTJ2LTFjMC0uMzQtLjA0LS42Ny0uMDktMUgyMFY4em0tNiA4aC00di0yaDR2MnptMC00aC00di0yaDR2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTYuMTdMNC44MyAxMmwtMS40MiAxLjQxTDkgMTkgMjEgN2wtMS40MS0xLjQxeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1pY29uLWJyYW5kMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNmZmYiPgogICAgPHBhdGggZD0iTTEwNSAxMjcuM2g0MHYxMi44aC00MHpNNTEuMSA3N0w3NCA5OS45bC0yMy4zIDIzLjMgMTAuNSAxMC41IDIzLjMtMjMuM0w5NSA5OS45IDg0LjUgODkuNCA2MS42IDY2LjV6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNGOUE4MjUiPgogICAgPHBhdGggZD0iTTIwLjIgMTEuOGMtMS42IDAtMS43LjUtMS43IDEgMCAuNC4xLjkuMSAxLjMuMS41LjEuOS4xIDEuMyAwIDEuNy0xLjQgMi4zLTMuNSAyLjNoLS45di0xLjloLjVjMS4xIDAgMS40IDAgMS40LS44IDAtLjMgMC0uNi0uMS0xIDAtLjQtLjEtLjgtLjEtMS4yIDAtMS4zIDAtMS44IDEuMy0yLTEuMy0uMi0xLjMtLjctMS4zLTIgMC0uNC4xLS44LjEtMS4yLjEtLjQuMS0uNy4xLTEgMC0uOC0uNC0uNy0xLjQtLjhoLS41VjQuMWguOWMyLjIgMCAzLjUuNyAzLjUgMi4zIDAgLjQtLjEuOS0uMSAxLjMtLjEuNS0uMS45LS4xIDEuMyAwIC41LjIgMSAxLjcgMXYxLjh6TTEuOCAxMC4xYzEuNiAwIDEuNy0uNSAxLjctMSAwLS40LS4xLS45LS4xLTEuMy0uMS0uNS0uMS0uOS0uMS0xLjMgMC0xLjYgMS40LTIuMyAzLjUtMi4zaC45djEuOWgtLjVjLTEgMC0xLjQgMC0xLjQuOCAwIC4zIDAgLjYuMSAxIDAgLjIuMS42LjEgMSAwIDEuMyAwIDEuOC0xLjMgMkM2IDExLjIgNiAxMS43IDYgMTNjMCAuNC0uMS44LS4xIDEuMi0uMS4zLS4xLjctLjEgMSAwIC44LjMuOCAxLjQuOGguNXYxLjloLS45Yy0yLjEgMC0zLjUtLjYtMy41LTIuMyAwLS40LjEtLjkuMS0xLjMuMS0uNS4xLS45LjEtMS4zIDAtLjUtLjItMS0xLjctMXYtMS45eiIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSIxMy44IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY3g9IjExIiBjeT0iOC4yIiByPSIyLjEiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgPGcgY2xhc3M9ImpwLWljb24td2FybjAiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4=);
  --jp-icon-listings-info: url(data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJDYXBhXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4Ig0KCSB2aWV3Qm94PSIwIDAgNTAuOTc4IDUwLjk3OCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgNTAuOTc4IDUwLjk3ODsiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxnPg0KCQkJPHBhdGggc3R5bGU9ImZpbGw6IzAxMDAwMjsiIGQ9Ik00My41Miw3LjQ1OEMzOC43MTEsMi42NDgsMzIuMzA3LDAsMjUuNDg5LDBDMTguNjcsMCwxMi4yNjYsMi42NDgsNy40NTgsNy40NTgNCgkJCQljLTkuOTQzLDkuOTQxLTkuOTQzLDI2LjExOSwwLDM2LjA2MmM0LjgwOSw0LjgwOSwxMS4yMTIsNy40NTYsMTguMDMxLDcuNDU4YzAsMCwwLjAwMSwwLDAuMDAyLDANCgkJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoNCgkJCQkgTTQyLjEwNiw0Mi4xMDVjLTQuNDMyLDQuNDMxLTEwLjMzMiw2Ljg3Mi0xNi42MTUsNi44NzJoLTAuMDAyYy02LjI4NS0wLjAwMS0xMi4xODctMi40NDEtMTYuNjE3LTYuODcyDQoJCQkJYy05LjE2Mi05LjE2My05LjE2Mi0yNC4wNzEsMC0zMy4yMzNDMTMuMzAzLDQuNDQsMTkuMjA0LDIsMjUuNDg5LDJjNi4yODQsMCwxMi4xODYsMi40NCwxNi42MTcsNi44NzINCgkJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4NCgkJPC9nPg0KCQk8Zz4NCgkJCTxwYXRoIHN0eWxlPSJmaWxsOiMwMTAwMDI7IiBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1Mw0KCQkJCWMwLjQ2OC0wLjUzNiwwLjkyMy0xLjA2MiwxLjM2Ny0xLjU3NWMwLjYyNi0wLjc1MywxLjEwNC0xLjQ3OCwxLjQzNi0yLjE3NWMwLjMzMS0wLjcwNywwLjQ5NS0xLjU0MSwwLjQ5NS0yLjUNCgkJCQljMC0xLjA5Ni0wLjI2LTIuMDg4LTAuNzc5LTIuOTc5Yy0wLjU2NS0wLjg3OS0xLjUwMS0xLjMzNi0yLjgwNi0xLjM2OWMtMS44MDIsMC4wNTctMi45ODUsMC42NjctMy41NSwxLjgzMg0KCQkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkNCgkJCQljMS4wNjItMS42NCwyLjg1NS0yLjQ4MSw1LjM3OC0yLjUyN2MyLjE2LDAuMDIzLDMuODc0LDAuNjA4LDUuMTQxLDEuNzU4YzEuMjc4LDEuMTYsMS45MjksMi43NjQsMS45NSw0LjgxMQ0KCQkJCWMwLDEuMTQyLTAuMTM3LDIuMTExLTAuNDEsMi45MTFjLTAuMzA5LDAuODQ1LTAuNzMxLDEuNTkzLTEuMjY4LDIuMjQzYy0wLjQ5MiwwLjY1LTEuMDY4LDEuMzE4LTEuNzMsMi4wMDINCgkJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5DQoJCQkJQzI2LjU4OSwzMi4yMTgsMjMuNTc4LDMyLjIxOCwyMy41NzgsMzIuMjE4eiBNMjMuNTc4LDM4LjIydi0zLjQ4NGgzLjA3NnYzLjQ4NEgyMy41Nzh6Ii8+DQoJCTwvZz4NCgk8L2c+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8L3N2Zz4NCg==);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMEQ0N0ExIj4KICAgIDxwYXRoIGQ9Ik0xMS4xIDYuOVY1LjhINi45YzAtLjUgMC0xLjMuMi0xLjYuNC0uNy44LTEuMSAxLjctMS40IDEuNy0uMyAyLjUtLjMgMy45LS4xIDEgLjEgMS45LjkgMS45IDEuOXY0LjJjMCAuNS0uOSAxLjYtMiAxLjZIOC44Yy0xLjUgMC0yLjQgMS40LTIuNCAyLjh2Mi4ySDQuN0MzLjUgMTUuMSAzIDE0IDMgMTMuMVY5Yy0uMS0xIC42LTIgMS44LTIgMS41LS4xIDYuMy0uMSA2LjMtLjF6Ii8+CiAgICA8cGF0aCBkPSJNMTAuOSAxNS4xdjEuMWg0LjJjMCAuNSAwIDEuMy0uMiAxLjYtLjQuNy0uOCAxLjEtMS43IDEuNC0xLjcuMy0yLjUuMy0zLjkuMS0xLS4xLTEuOS0uOS0xLjktMS45di00LjJjMC0uNS45LTEuNiAyLTEuNmgzLjhjMS41IDAgMi40LTEuNCAyLjQtMi44VjYuNmgxLjdDMTguNSA2LjkgMTkgOCAxOSA4LjlWMTNjMCAxLS43IDIuMS0xLjkgMi4xaC02LjJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMikiIGZpbGw9IiMzMzMzMzMiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uLWFjY2VudDIganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGQ9Ik01LjA1NjY0IDguNzYxNzJDNS4wNTY2NCA4LjU5NzY2IDUuMDMxMjUgOC40NTMxMiA0Ljk4MDQ3IDguMzI4MTJDNC45MzM1OSA4LjE5OTIyIDQuODU1NDcgOC4wODIwMyA0Ljc0NjA5IDcuOTc2NTZDNC42NDA2MiA3Ljg3MTA5IDQuNSA3Ljc3NTM5IDQuMzI0MjIgNy42ODk0NUM0LjE1MjM0IDcuNTk5NjEgMy45NDMzNiA3LjUxMTcyIDMuNjk3MjcgNy40MjU3OEMzLjMwMjczIDcuMjg1MTYgMi45NDMzNiA3LjEzNjcyIDIuNjE5MTQgNi45ODA0N0MyLjI5NDkyIDYuODI0MjIgMi4wMTc1OCA2LjY0MjU4IDEuNzg3MTEgNi40MzU1NUMxLjU2MDU1IDYuMjI4NTIgMS4zODQ3NyA1Ljk4ODI4IDEuMjU5NzcgNS43MTQ4NEMxLjEzNDc3IDUuNDM3NSAxLjA3MjI3IDUuMTA5MzggMS4wNzIyNyA0LjczMDQ3QzEuMDcyMjcgNC4zOTg0NCAxLjEyODkxIDQuMDk1NyAxLjI0MjE5IDMuODIyMjdDMS4zNTU0NyAzLjU0NDkyIDEuNTE1NjIgMy4zMDQ2OSAxLjcyMjY2IDMuMTAxNTZDMS45Mjk2OSAyLjg5ODQ0IDIuMTc5NjkgMi43MzQzNyAyLjQ3MjY2IDIuNjA5MzhDMi43NjU2MiAyLjQ4NDM4IDMuMDkxOCAyLjQwNDMgMy40NTExNyAyLjM2OTE0VjEuMTA5MzhINC4zODg2N1YyLjM4MDg2QzQuNzQwMjMgMi40Mjc3MyA1LjA1NjY0IDIuNTIzNDQgNS4zMzc4OSAyLjY2Nzk3QzUuNjE5MTQgMi44MTI1IDUuODU3NDIgMy4wMDE5NSA2LjA1MjczIDMuMjM2MzNDNi4yNTE5NSAzLjQ2NjggNi40MDQzIDMuNzQwMjMgNi41MDk3NyA0LjA1NjY0QzYuNjE5MTQgNC4zNjkxNCA2LjY3MzgzIDQuNzIwNyA2LjY3MzgzIDUuMTExMzNINS4wNDQ5MkM1LjA0NDkyIDQuNjM4NjcgNC45Mzc1IDQuMjgxMjUgNC43MjI2NiA0LjAzOTA2QzQuNTA3ODEgMy43OTI5NyA0LjIxNjggMy42Njk5MiAzLjg0OTYxIDMuNjY5OTJDMy42NTAzOSAzLjY2OTkyIDMuNDc2NTYgMy42OTcyNyAzLjMyODEyIDMuNzUxOTVDMy4xODM1OSAzLjgwMjczIDMuMDY0NDUgMy44NzY5NSAyLjk3MDcgMy45NzQ2MUMyLjg3Njk1IDQuMDY4MzYgMi44MDY2NCA0LjE3OTY5IDIuNzU5NzcgNC4zMDg1OUMyLjcxNjggNC40Mzc1IDIuNjk1MzEgNC41NzgxMiAyLjY5NTMxIDQuNzMwNDdDMi42OTUzMSA0Ljg4MjgxIDIuNzE2OCA1LjAxOTUzIDIuNzU5NzcgNS4xNDA2MkMyLjgwNjY0IDUuMjU3ODEgMi44ODI4MSA1LjM2NzE5IDIuOTg4MjggNS40Njg3NUMzLjA5NzY2IDUuNTcwMzEgMy4yNDAyMyA1LjY2Nzk3IDMuNDE2MDIgNS43NjE3MkMzLjU5MTggNS44NTE1NiAzLjgxMDU1IDUuOTQzMzYgNC4wNzIyNyA2LjAzNzExQzQuNDY2OCA2LjE4NTU1IDQuODI0MjIgNi4zMzk4NCA1LjE0NDUzIDYuNUM1LjQ2NDg0IDYuNjU2MjUgNS43MzgyOCA2LjgzOTg0IDUuOTY0ODQgNy4wNTA3OEM2LjE5NTMxIDcuMjU3ODEgNi4zNzEwOSA3LjUgNi40OTIxOSA3Ljc3NzM0QzYuNjE3MTkgOC4wNTA3OCA2LjY3OTY5IDguMzc1IDYuNjc5NjkgOC43NUM2LjY3OTY5IDkuMDkzNzUgNi42MjMwNSA5LjQwNDMgNi41MDk3NyA5LjY4MTY0QzYuMzk2NDggOS45NTUwOCA2LjIzNDM4IDEwLjE5MTQgNi4wMjM0NCAxMC4zOTA2QzUuODEyNSAxMC41ODk4IDUuNTU4NTkgMTAuNzUgNS4yNjE3MiAxMC44NzExQzQuOTY0ODQgMTAuOTg4MyA0LjYzMjgxIDExLjA2NDUgNC4yNjU2MiAxMS4wOTk2VjEyLjI0OEgzLjMzMzk4VjExLjA5OTZDMy4wMDE5NSAxMS4wNjg0IDIuNjc5NjkgMTAuOTk2MSAyLjM2NzE5IDEwLjg4MjhDMi4wNTQ2OSAxMC43NjU2IDEuNzc3MzQgMTAuNTk3NyAxLjUzNTE2IDEwLjM3ODlDMS4yOTY4OCAxMC4xNjAyIDEuMTA1NDcgOS44ODQ3NyAwLjk2MDkzOCA5LjU1MjczQzAuODE2NDA2IDkuMjE2OCAwLjc0NDE0MSA4LjgxNDQ1IDAuNzQ0MTQxIDguMzQ1N0gyLjM3ODkxQzIuMzc4OTEgOC42MjY5NSAyLjQxOTkyIDguODYzMjggMi41MDE5NSA5LjA1NDY5QzIuNTgzOTggOS4yNDIxOSAyLjY4OTQ1IDkuMzkyNTggMi44MTgzNiA5LjUwNTg2QzIuOTUxMTcgOS42MTUyMyAzLjEwMTU2IDkuNjkzMzYgMy4yNjk1MyA5Ljc0MDIzQzMuNDM3NSA5Ljc4NzExIDMuNjA5MzggOS44MTA1NSAzLjc4NTE2IDkuODEwNTVDNC4yMDMxMiA5LjgxMDU1IDQuNTE5NTMgOS43MTI4OSA0LjczNDM4IDkuNTE3NThDNC45NDkyMiA5LjMyMjI3IDUuMDU2NjQgOS4wNzAzMSA1LjA1NjY0IDguNzYxNzJaTTEzLjQxOCAxMi4yNzE1SDguMDc0MjJWMTFIMTMuNDE4VjEyLjI3MTVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzLjk1MjY0IDYpIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTUgMTVIM3YyaDEydi0yem0wLThIM3YyaDEyVjd6TTMgMTNoMTh2LTJIM3Yyem0wIDhoMTh2LTJIM3Yyek0zIDN2MmgxOFYzSDN6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}
.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}
.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}
.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}
.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}
.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}
.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}
.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}
.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}
.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}
.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}
.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}
.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}
.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}
.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}
.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}
.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}
.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}
.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}
.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}
.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}
.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}
.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}
.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}
.jp-FileIcon {
  background-image: var(--jp-icon-file);
}
.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}
.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}
.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}
.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}
.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}
.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}
.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}
.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}
.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}
.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}
.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}
.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}
.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}
.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}
.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}
.jp-ListIcon {
  background-image: var(--jp-icon-list);
}
.jp-ListingsInfoIcon {
  background-image: var(--jp-icon-listings-info);
}
.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}
.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}
.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}
.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}
.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}
.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}
.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}
.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}
.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}
.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}
.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}
.jp-RunIcon {
  background-image: var(--jp-icon-run);
}
.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}
.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}
.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}
.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}
.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}
.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}
.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}
.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}
.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}
.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}
.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}
.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}
.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

:root {
  --jp-icon-search-white: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
}

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}
/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}
/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}
/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}
.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}
.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}
.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}
.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}
.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}
.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}
.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}
.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}
/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}
.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}
.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}
.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}
.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}
.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}
.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}
/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

/* CSS for icons in selected items in the settings editor */
#setting-editor .jp-PluginList .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
#setting-editor
  .jp-PluginList
  .jp-mod-selected
  .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected tabs in the sidebar tab manager */
#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable[fill] {
  fill: #fff;
}

#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable[fill] {
  fill: var(--jp-brand-color1);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable-inverse[fill] {
  fill: #fff;
}

/**
 * TODO: come up with non css-hack solution for showing the busy icon on top
 *  of the close icon
 * CSS for complex behavior of close icon of tabs in the sidebar tab manager
 */
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-dirty.jp-mod-active
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: #fff;
}

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) svg {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

/* Override Blueprint's _reset.scss styles */
html {
  box-sizing: unset;
}

*,
*::before,
*::after {
  box-sizing: unset;
}

body {
  color: unset;
  font-family: var(--jp-ui-font-family);
}

p {
  margin-top: unset;
  margin-bottom: unset;
}

small {
  font-size: unset;
}

strong {
  font-weight: unset;
}

/* Override Blueprint's _typography.scss styles */
a {
  text-decoration: unset;
  color: unset;
}
a:hover {
  text-decoration: unset;
  color: unset;
}

/* Override Blueprint's _accessibility.scss styles */
:focus {
  outline: unset;
  outline-offset: unset;
  -moz-outline-radius: unset;
}

/* Styles for ui-components */
.jp-Button {
  border-radius: var(--jp-border-radius);
  padding: 0px 12px;
  font-size: var(--jp-ui-font-size1);
}

/* Use our own theme for hover styles */
button.jp-Button.bp3-button.bp3-minimal:hover {
  background-color: var(--jp-layout-color2);
}
.jp-Button.minimal {
  color: unset !important;
}

.jp-Button.jp-ToolbarButtonComponent {
  text-transform: none;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color3);
}

.jp-BPIcon {
  display: inline-block;
  vertical-align: middle;
  margin: auto;
}

/* Stop blueprint futzing with our icon fills */
.bp3-icon.jp-BPIcon > svg:not([fill]) {
  fill: var(--jp-inverse-layout-color3);
}

.jp-InputGroupAction {
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

/* Use our own theme for hover and option styles */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}
select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-top: 1px solid var(--jp-border-color2);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-Collapse-header {
  padding: 1px 12px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size2);
}

.jp-Collapse-header:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Collapse-contents {
  padding: 0px 12px 0px 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0px;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0px 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.lm-CommandPalette-wrapper::after {
  content: ' ';
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  height: 30px;
  width: 10px;
  padding: 0px 10px;
  background-image: var(--jp-icon-search-white);
  background-size: 20px;
  background-repeat: no-repeat;
  background-position: center;
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color3);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0px;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color3);
}

.lm-CommandPalette-item.lm-mod-active {
  background: var(--jp-layout-color3);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  background: var(--jp-layout-color4);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color3);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.4;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty:after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0px 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0px;
  left: 0px;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px;
  padding-bottom: 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0px;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

.jp-Dialog-header {
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0px 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

.jp-HoverBox.jp-mod-outofview {
  display: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;

  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;

  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #aa00ff;

  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;

  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;

  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;

  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;

  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;

  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;

  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;

  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;

  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;

  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ffff00;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;

  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;

  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;

  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;

  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;

  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eeeeee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;

  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent:before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent:after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0px 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  height: 28px;
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  background-color: var(--jp-layout-color1);
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0px 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  height: 32px;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 1;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px;
  margin: 0px;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px 6px;
  margin: 0px;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent span {
  padding: 0px;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ body.p-mod-override-cursor *, /* </DEPRECATED> */
body.lm-mod-override-cursor * {
  cursor: inherit !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* BASICS */

.CodeMirror {
  /* Set height, width, borders, and global font properties here */
  font-family: monospace;
  height: 300px;
  color: black;
  direction: ltr;
}

/* PADDING */

.CodeMirror-lines {
  padding: 4px 0; /* Vertical padding around content */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  padding: 0 4px; /* Horizontal padding of content */
}

.CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  background-color: white; /* The little square between H and V scrollbars */
}

/* GUTTER */

.CodeMirror-gutters {
  border-right: 1px solid #ddd;
  background-color: #f7f7f7;
  white-space: nowrap;
}
.CodeMirror-linenumbers {}
.CodeMirror-linenumber {
  padding: 0 3px 0 5px;
  min-width: 20px;
  text-align: right;
  color: #999;
  white-space: nowrap;
}

.CodeMirror-guttermarker { color: black; }
.CodeMirror-guttermarker-subtle { color: #999; }

/* CURSOR */

.CodeMirror-cursor {
  border-left: 1px solid black;
  border-right: none;
  width: 0;
}
/* Shown when moving in bi-directional text */
.CodeMirror div.CodeMirror-secondarycursor {
  border-left: 1px solid silver;
}
.cm-fat-cursor .CodeMirror-cursor {
  width: auto;
  border: 0 !important;
  background: #7e7;
}
.cm-fat-cursor div.CodeMirror-cursors {
  z-index: 1;
}
.cm-fat-cursor-mark {
  background-color: rgba(20, 255, 20, 0.5);
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
}
.cm-animate-fat-cursor {
  width: auto;
  border: 0;
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
  background-color: #7e7;
}
@-moz-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@-webkit-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}

/* Can style cursor different in overwrite (non-insert) mode */
.CodeMirror-overwrite .CodeMirror-cursor {}

.cm-tab { display: inline-block; text-decoration: inherit; }

.CodeMirror-rulers {
  position: absolute;
  left: 0; right: 0; top: -50px; bottom: 0;
  overflow: hidden;
}
.CodeMirror-ruler {
  border-left: 1px solid #ccc;
  top: 0; bottom: 0;
  position: absolute;
}

/* DEFAULT THEME */

.cm-s-default .cm-header {color: blue;}
.cm-s-default .cm-quote {color: #090;}
.cm-negative {color: #d44;}
.cm-positive {color: #292;}
.cm-header, .cm-strong {font-weight: bold;}
.cm-em {font-style: italic;}
.cm-link {text-decoration: underline;}
.cm-strikethrough {text-decoration: line-through;}

.cm-s-default .cm-keyword {color: #708;}
.cm-s-default .cm-atom {color: #219;}
.cm-s-default .cm-number {color: #164;}
.cm-s-default .cm-def {color: #00f;}
.cm-s-default .cm-variable,
.cm-s-default .cm-punctuation,
.cm-s-default .cm-property,
.cm-s-default .cm-operator {}
.cm-s-default .cm-variable-2 {color: #05a;}
.cm-s-default .cm-variable-3, .cm-s-default .cm-type {color: #085;}
.cm-s-default .cm-comment {color: #a50;}
.cm-s-default .cm-string {color: #a11;}
.cm-s-default .cm-string-2 {color: #f50;}
.cm-s-default .cm-meta {color: #555;}
.cm-s-default .cm-qualifier {color: #555;}
.cm-s-default .cm-builtin {color: #30a;}
.cm-s-default .cm-bracket {color: #997;}
.cm-s-default .cm-tag {color: #170;}
.cm-s-default .cm-attribute {color: #00c;}
.cm-s-default .cm-hr {color: #999;}
.cm-s-default .cm-link {color: #00c;}

.cm-s-default .cm-error {color: #f00;}
.cm-invalidchar {color: #f00;}

.CodeMirror-composing { border-bottom: 2px solid; }

/* Default styles for common addons */

div.CodeMirror span.CodeMirror-matchingbracket {color: #0b0;}
div.CodeMirror span.CodeMirror-nonmatchingbracket {color: #a22;}
.CodeMirror-matchingtag { background: rgba(255, 150, 0, .3); }
.CodeMirror-activeline-background {background: #e8f2ff;}

/* STOP */

/* The rest of this file contains styles related to the mechanics of
   the editor. You probably shouldn't touch them. */

.CodeMirror {
  position: relative;
  overflow: hidden;
  background: white;
}

.CodeMirror-scroll {
  overflow: scroll !important; /* Things will break if this is overridden */
  /* 30px is the magic margin used to hide the element's real scrollbars */
  /* See overflow: hidden in .CodeMirror */
  margin-bottom: -30px; margin-right: -30px;
  padding-bottom: 30px;
  height: 100%;
  outline: none; /* Prevent dragging from highlighting the element */
  position: relative;
}
.CodeMirror-sizer {
  position: relative;
  border-right: 30px solid transparent;
}

/* The fake, visible scrollbars. Used to force redraw during scrolling
   before actual scrolling happens, thus preventing shaking and
   flickering artifacts. */
.CodeMirror-vscrollbar, .CodeMirror-hscrollbar, .CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  position: absolute;
  z-index: 6;
  display: none;
}
.CodeMirror-vscrollbar {
  right: 0; top: 0;
  overflow-x: hidden;
  overflow-y: scroll;
}
.CodeMirror-hscrollbar {
  bottom: 0; left: 0;
  overflow-y: hidden;
  overflow-x: scroll;
}
.CodeMirror-scrollbar-filler {
  right: 0; bottom: 0;
}
.CodeMirror-gutter-filler {
  left: 0; bottom: 0;
}

.CodeMirror-gutters {
  position: absolute; left: 0; top: 0;
  min-height: 100%;
  z-index: 3;
}
.CodeMirror-gutter {
  white-space: normal;
  height: 100%;
  display: inline-block;
  vertical-align: top;
  margin-bottom: -30px;
}
.CodeMirror-gutter-wrapper {
  position: absolute;
  z-index: 4;
  background: none !important;
  border: none !important;
}
.CodeMirror-gutter-background {
  position: absolute;
  top: 0; bottom: 0;
  z-index: 4;
}
.CodeMirror-gutter-elt {
  position: absolute;
  cursor: default;
  z-index: 4;
}
.CodeMirror-gutter-wrapper ::selection { background-color: transparent }
.CodeMirror-gutter-wrapper ::-moz-selection { background-color: transparent }

.CodeMirror-lines {
  cursor: text;
  min-height: 1px; /* prevents collapsing before first draw */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  /* Reset some styles that the rest of the page might have set */
  -moz-border-radius: 0; -webkit-border-radius: 0; border-radius: 0;
  border-width: 0;
  background: transparent;
  font-family: inherit;
  font-size: inherit;
  margin: 0;
  white-space: pre;
  word-wrap: normal;
  line-height: inherit;
  color: inherit;
  z-index: 2;
  position: relative;
  overflow: visible;
  -webkit-tap-highlight-color: transparent;
  -webkit-font-variant-ligatures: contextual;
  font-variant-ligatures: contextual;
}
.CodeMirror-wrap pre.CodeMirror-line,
.CodeMirror-wrap pre.CodeMirror-line-like {
  word-wrap: break-word;
  white-space: pre-wrap;
  word-break: normal;
}

.CodeMirror-linebackground {
  position: absolute;
  left: 0; right: 0; top: 0; bottom: 0;
  z-index: 0;
}

.CodeMirror-linewidget {
  position: relative;
  z-index: 2;
  padding: 0.1px; /* Force widget margins to stay inside of the container */
}

.CodeMirror-widget {}

.CodeMirror-rtl pre { direction: rtl; }

.CodeMirror-code {
  outline: none;
}

/* Force content-box sizing for the elements where we expect it */
.CodeMirror-scroll,
.CodeMirror-sizer,
.CodeMirror-gutter,
.CodeMirror-gutters,
.CodeMirror-linenumber {
  -moz-box-sizing: content-box;
  box-sizing: content-box;
}

.CodeMirror-measure {
  position: absolute;
  width: 100%;
  height: 0;
  overflow: hidden;
  visibility: hidden;
}

.CodeMirror-cursor {
  position: absolute;
  pointer-events: none;
}
.CodeMirror-measure pre { position: static; }

div.CodeMirror-cursors {
  visibility: hidden;
  position: relative;
  z-index: 3;
}
div.CodeMirror-dragcursors {
  visibility: visible;
}

.CodeMirror-focused div.CodeMirror-cursors {
  visibility: visible;
}

.CodeMirror-selected { background: #d9d9d9; }
.CodeMirror-focused .CodeMirror-selected { background: #d7d4f0; }
.CodeMirror-crosshair { cursor: crosshair; }
.CodeMirror-line::selection, .CodeMirror-line > span::selection, .CodeMirror-line > span > span::selection { background: #d7d4f0; }
.CodeMirror-line::-moz-selection, .CodeMirror-line > span::-moz-selection, .CodeMirror-line > span > span::-moz-selection { background: #d7d4f0; }

.cm-searching {
  background-color: #ffa;
  background-color: rgba(255, 255, 0, .4);
}

/* Used to force a border model for a node */
.cm-force-border { padding-right: .1px; }

@media print {
  /* Hide the cursor when printing */
  .CodeMirror div.CodeMirror-cursors {
    visibility: hidden;
  }
}

/* See issue #2901 */
.cm-tab-wrap-hack:after { content: ''; }

/* Help users use markselection to safely style text background */
span.CodeMirror-selectedtext { background: none; }

.CodeMirror-dialog {
  position: absolute;
  left: 0; right: 0;
  background: inherit;
  z-index: 15;
  padding: .1em .8em;
  overflow: hidden;
  color: inherit;
}

.CodeMirror-dialog-top {
  border-bottom: 1px solid #eee;
  top: 0;
}

.CodeMirror-dialog-bottom {
  border-top: 1px solid #eee;
  bottom: 0;
}

.CodeMirror-dialog input {
  border: none;
  outline: none;
  background: transparent;
  width: 20em;
  color: inherit;
  font-family: monospace;
}

.CodeMirror-dialog button {
  font-size: 70%;
}

.CodeMirror-foldmarker {
  color: blue;
  text-shadow: #b9f 1px 1px 2px, #b9f -1px -1px 2px, #b9f 1px -1px 2px, #b9f -1px 1px 2px;
  font-family: arial;
  line-height: .3;
  cursor: pointer;
}
.CodeMirror-foldgutter {
  width: .7em;
}
.CodeMirror-foldgutter-open,
.CodeMirror-foldgutter-folded {
  cursor: pointer;
}
.CodeMirror-foldgutter-open:after {
  content: "\25BE";
}
.CodeMirror-foldgutter-folded:after {
  content: "\25B8";
}

/*
  Name:       material
  Author:     Mattia Astorino (http://github.com/equinusocio)
  Website:    https://material-theme.site/
*/

.cm-s-material.CodeMirror {
  background-color: #263238;
  color: #EEFFFF;
}

.cm-s-material .CodeMirror-gutters {
  background: #263238;
  color: #546E7A;
  border: none;
}

.cm-s-material .CodeMirror-guttermarker,
.cm-s-material .CodeMirror-guttermarker-subtle,
.cm-s-material .CodeMirror-linenumber {
  color: #546E7A;
}

.cm-s-material .CodeMirror-cursor {
  border-left: 1px solid #FFCC00;
}

.cm-s-material div.CodeMirror-selected {
  background: rgba(128, 203, 196, 0.2);
}

.cm-s-material.CodeMirror-focused div.CodeMirror-selected {
  background: rgba(128, 203, 196, 0.2);
}

.cm-s-material .CodeMirror-line::selection,
.cm-s-material .CodeMirror-line>span::selection,
.cm-s-material .CodeMirror-line>span>span::selection {
  background: rgba(128, 203, 196, 0.2);
}

.cm-s-material .CodeMirror-line::-moz-selection,
.cm-s-material .CodeMirror-line>span::-moz-selection,
.cm-s-material .CodeMirror-line>span>span::-moz-selection {
  background: rgba(128, 203, 196, 0.2);
}

.cm-s-material .CodeMirror-activeline-background {
  background: rgba(0, 0, 0, 0.5);
}

.cm-s-material .cm-keyword {
  color: #C792EA;
}

.cm-s-material .cm-operator {
  color: #89DDFF;
}

.cm-s-material .cm-variable-2 {
  color: #EEFFFF;
}

.cm-s-material .cm-variable-3,
.cm-s-material .cm-type {
  color: #f07178;
}

.cm-s-material .cm-builtin {
  color: #FFCB6B;
}

.cm-s-material .cm-atom {
  color: #F78C6C;
}

.cm-s-material .cm-number {
  color: #FF5370;
}

.cm-s-material .cm-def {
  color: #82AAFF;
}

.cm-s-material .cm-string {
  color: #C3E88D;
}

.cm-s-material .cm-string-2 {
  color: #f07178;
}

.cm-s-material .cm-comment {
  color: #546E7A;
}

.cm-s-material .cm-variable {
  color: #f07178;
}

.cm-s-material .cm-tag {
  color: #FF5370;
}

.cm-s-material .cm-meta {
  color: #FFCB6B;
}

.cm-s-material .cm-attribute {
  color: #C792EA;
}

.cm-s-material .cm-property {
  color: #C792EA;
}

.cm-s-material .cm-qualifier {
  color: #DECB6B;
}

.cm-s-material .cm-variable-3,
.cm-s-material .cm-type {
  color: #DECB6B;
}


.cm-s-material .cm-error {
  color: rgba(255, 255, 255, 1.0);
  background-color: #FF5370;
}

.cm-s-material .CodeMirror-matchingbracket {
  text-decoration: underline;
  color: white !important;
}
/**
 * "
 *  Using Zenburn color palette from the Emacs Zenburn Theme
 *  https://github.com/bbatsov/zenburn-emacs/blob/master/zenburn-theme.el
 *
 *  Also using parts of https://github.com/xavi/coderay-lighttable-theme
 * "
 * From: https://github.com/wisenomad/zenburn-lighttable-theme/blob/master/zenburn.css
 */

.cm-s-zenburn .CodeMirror-gutters { background: #3f3f3f !important; }
.cm-s-zenburn .CodeMirror-foldgutter-open, .CodeMirror-foldgutter-folded { color: #999; }
.cm-s-zenburn .CodeMirror-cursor { border-left: 1px solid white; }
.cm-s-zenburn { background-color: #3f3f3f; color: #dcdccc; }
.cm-s-zenburn span.cm-builtin { color: #dcdccc; font-weight: bold; }
.cm-s-zenburn span.cm-comment { color: #7f9f7f; }
.cm-s-zenburn span.cm-keyword { color: #f0dfaf; font-weight: bold; }
.cm-s-zenburn span.cm-atom { color: #bfebbf; }
.cm-s-zenburn span.cm-def { color: #dcdccc; }
.cm-s-zenburn span.cm-variable { color: #dfaf8f; }
.cm-s-zenburn span.cm-variable-2 { color: #dcdccc; }
.cm-s-zenburn span.cm-string { color: #cc9393; }
.cm-s-zenburn span.cm-string-2 { color: #cc9393; }
.cm-s-zenburn span.cm-number { color: #dcdccc; }
.cm-s-zenburn span.cm-tag { color: #93e0e3; }
.cm-s-zenburn span.cm-property { color: #dfaf8f; }
.cm-s-zenburn span.cm-attribute { color: #dfaf8f; }
.cm-s-zenburn span.cm-qualifier { color: #7cb8bb; }
.cm-s-zenburn span.cm-meta { color: #f0dfaf; }
.cm-s-zenburn span.cm-header { color: #f0efd0; }
.cm-s-zenburn span.cm-operator { color: #f0efd0; }
.cm-s-zenburn span.CodeMirror-matchingbracket { box-sizing: border-box; background: transparent; border-bottom: 1px solid; }
.cm-s-zenburn span.CodeMirror-nonmatchingbracket { border-bottom: 1px solid; background: none; }
.cm-s-zenburn .CodeMirror-activeline { background: #000000; }
.cm-s-zenburn .CodeMirror-activeline-background { background: #000000; }
.cm-s-zenburn div.CodeMirror-selected { background: #545454; }
.cm-s-zenburn .CodeMirror-focused div.CodeMirror-selected { background: #4f4f4f; }

.cm-s-abcdef.CodeMirror { background: #0f0f0f; color: #defdef; }
.cm-s-abcdef div.CodeMirror-selected { background: #515151; }
.cm-s-abcdef .CodeMirror-line::selection, .cm-s-abcdef .CodeMirror-line > span::selection, .cm-s-abcdef .CodeMirror-line > span > span::selection { background: rgba(56, 56, 56, 0.99); }
.cm-s-abcdef .CodeMirror-line::-moz-selection, .cm-s-abcdef .CodeMirror-line > span::-moz-selection, .cm-s-abcdef .CodeMirror-line > span > span::-moz-selection { background: rgba(56, 56, 56, 0.99); }
.cm-s-abcdef .CodeMirror-gutters { background: #555; border-right: 2px solid #314151; }
.cm-s-abcdef .CodeMirror-guttermarker { color: #222; }
.cm-s-abcdef .CodeMirror-guttermarker-subtle { color: azure; }
.cm-s-abcdef .CodeMirror-linenumber { color: #FFFFFF; }
.cm-s-abcdef .CodeMirror-cursor { border-left: 1px solid #00FF00; }

.cm-s-abcdef span.cm-keyword { color: darkgoldenrod; font-weight: bold; }
.cm-s-abcdef span.cm-atom { color: #77F; }
.cm-s-abcdef span.cm-number { color: violet; }
.cm-s-abcdef span.cm-def { color: #fffabc; }
.cm-s-abcdef span.cm-variable { color: #abcdef; }
.cm-s-abcdef span.cm-variable-2 { color: #cacbcc; }
.cm-s-abcdef span.cm-variable-3, .cm-s-abcdef span.cm-type { color: #def; }
.cm-s-abcdef span.cm-property { color: #fedcba; }
.cm-s-abcdef span.cm-operator { color: #ff0; }
.cm-s-abcdef span.cm-comment { color: #7a7b7c; font-style: italic;}
.cm-s-abcdef span.cm-string { color: #2b4; }
.cm-s-abcdef span.cm-meta { color: #C9F; }
.cm-s-abcdef span.cm-qualifier { color: #FFF700; }
.cm-s-abcdef span.cm-builtin { color: #30aabc; }
.cm-s-abcdef span.cm-bracket { color: #8a8a8a; }
.cm-s-abcdef span.cm-tag { color: #FFDD44; }
.cm-s-abcdef span.cm-attribute { color: #DDFF00; }
.cm-s-abcdef span.cm-error { color: #FF0000; }
.cm-s-abcdef span.cm-header { color: aquamarine; font-weight: bold; }
.cm-s-abcdef span.cm-link { color: blueviolet; }

.cm-s-abcdef .CodeMirror-activeline-background { background: #314151; }

/*

    Name:       Base16 Default Light
    Author:     Chris Kempson (http://chriskempson.com)

    CodeMirror template by Jan T. Sott (https://github.com/idleberg/base16-codemirror)
    Original Base16 color scheme by Chris Kempson (https://github.com/chriskempson/base16)

*/

.cm-s-base16-light.CodeMirror { background: #f5f5f5; color: #202020; }
.cm-s-base16-light div.CodeMirror-selected { background: #e0e0e0; }
.cm-s-base16-light .CodeMirror-line::selection, .cm-s-base16-light .CodeMirror-line > span::selection, .cm-s-base16-light .CodeMirror-line > span > span::selection { background: #e0e0e0; }
.cm-s-base16-light .CodeMirror-line::-moz-selection, .cm-s-base16-light .CodeMirror-line > span::-moz-selection, .cm-s-base16-light .CodeMirror-line > span > span::-moz-selection { background: #e0e0e0; }
.cm-s-base16-light .CodeMirror-gutters { background: #f5f5f5; border-right: 0px; }
.cm-s-base16-light .CodeMirror-guttermarker { color: #ac4142; }
.cm-s-base16-light .CodeMirror-guttermarker-subtle { color: #b0b0b0; }
.cm-s-base16-light .CodeMirror-linenumber { color: #b0b0b0; }
.cm-s-base16-light .CodeMirror-cursor { border-left: 1px solid #505050; }

.cm-s-base16-light span.cm-comment { color: #8f5536; }
.cm-s-base16-light span.cm-atom { color: #aa759f; }
.cm-s-base16-light span.cm-number { color: #aa759f; }

.cm-s-base16-light span.cm-property, .cm-s-base16-light span.cm-attribute { color: #90a959; }
.cm-s-base16-light span.cm-keyword { color: #ac4142; }
.cm-s-base16-light span.cm-string { color: #f4bf75; }

.cm-s-base16-light span.cm-variable { color: #90a959; }
.cm-s-base16-light span.cm-variable-2 { color: #6a9fb5; }
.cm-s-base16-light span.cm-def { color: #d28445; }
.cm-s-base16-light span.cm-bracket { color: #202020; }
.cm-s-base16-light span.cm-tag { color: #ac4142; }
.cm-s-base16-light span.cm-link { color: #aa759f; }
.cm-s-base16-light span.cm-error { background: #ac4142; color: #505050; }

.cm-s-base16-light .CodeMirror-activeline-background { background: #DDDCDC; }
.cm-s-base16-light .CodeMirror-matchingbracket { color: #f5f5f5 !important; background-color: #6A9FB5 !important}

/*

    Name:       Base16 Default Dark
    Author:     Chris Kempson (http://chriskempson.com)

    CodeMirror template by Jan T. Sott (https://github.com/idleberg/base16-codemirror)
    Original Base16 color scheme by Chris Kempson (https://github.com/chriskempson/base16)

*/

.cm-s-base16-dark.CodeMirror { background: #151515; color: #e0e0e0; }
.cm-s-base16-dark div.CodeMirror-selected { background: #303030; }
.cm-s-base16-dark .CodeMirror-line::selection, .cm-s-base16-dark .CodeMirror-line > span::selection, .cm-s-base16-dark .CodeMirror-line > span > span::selection { background: rgba(48, 48, 48, .99); }
.cm-s-base16-dark .CodeMirror-line::-moz-selection, .cm-s-base16-dark .CodeMirror-line > span::-moz-selection, .cm-s-base16-dark .CodeMirror-line > span > span::-moz-selection { background: rgba(48, 48, 48, .99); }
.cm-s-base16-dark .CodeMirror-gutters { background: #151515; border-right: 0px; }
.cm-s-base16-dark .CodeMirror-guttermarker { color: #ac4142; }
.cm-s-base16-dark .CodeMirror-guttermarker-subtle { color: #505050; }
.cm-s-base16-dark .CodeMirror-linenumber { color: #505050; }
.cm-s-base16-dark .CodeMirror-cursor { border-left: 1px solid #b0b0b0; }

.cm-s-base16-dark span.cm-comment { color: #8f5536; }
.cm-s-base16-dark span.cm-atom { color: #aa759f; }
.cm-s-base16-dark span.cm-number { color: #aa759f; }

.cm-s-base16-dark span.cm-property, .cm-s-base16-dark span.cm-attribute { color: #90a959; }
.cm-s-base16-dark span.cm-keyword { color: #ac4142; }
.cm-s-base16-dark span.cm-string { color: #f4bf75; }

.cm-s-base16-dark span.cm-variable { color: #90a959; }
.cm-s-base16-dark span.cm-variable-2 { color: #6a9fb5; }
.cm-s-base16-dark span.cm-def { color: #d28445; }
.cm-s-base16-dark span.cm-bracket { color: #e0e0e0; }
.cm-s-base16-dark span.cm-tag { color: #ac4142; }
.cm-s-base16-dark span.cm-link { color: #aa759f; }
.cm-s-base16-dark span.cm-error { background: #ac4142; color: #b0b0b0; }

.cm-s-base16-dark .CodeMirror-activeline-background { background: #202020; }
.cm-s-base16-dark .CodeMirror-matchingbracket { text-decoration: underline; color: white !important; }

/*

    Name:       dracula
    Author:     Michael Kaminsky (http://github.com/mkaminsky11)

    Original dracula color scheme by Zeno Rocha (https://github.com/zenorocha/dracula-theme)

*/


.cm-s-dracula.CodeMirror, .cm-s-dracula .CodeMirror-gutters {
  background-color: #282a36 !important;
  color: #f8f8f2 !important;
  border: none;
}
.cm-s-dracula .CodeMirror-gutters { color: #282a36; }
.cm-s-dracula .CodeMirror-cursor { border-left: solid thin #f8f8f0; }
.cm-s-dracula .CodeMirror-linenumber { color: #6D8A88; }
.cm-s-dracula .CodeMirror-selected { background: rgba(255, 255, 255, 0.10); }
.cm-s-dracula .CodeMirror-line::selection, .cm-s-dracula .CodeMirror-line > span::selection, .cm-s-dracula .CodeMirror-line > span > span::selection { background: rgba(255, 255, 255, 0.10); }
.cm-s-dracula .CodeMirror-line::-moz-selection, .cm-s-dracula .CodeMirror-line > span::-moz-selection, .cm-s-dracula .CodeMirror-line > span > span::-moz-selection { background: rgba(255, 255, 255, 0.10); }
.cm-s-dracula span.cm-comment { color: #6272a4; }
.cm-s-dracula span.cm-string, .cm-s-dracula span.cm-string-2 { color: #f1fa8c; }
.cm-s-dracula span.cm-number { color: #bd93f9; }
.cm-s-dracula span.cm-variable { color: #50fa7b; }
.cm-s-dracula span.cm-variable-2 { color: white; }
.cm-s-dracula span.cm-def { color: #50fa7b; }
.cm-s-dracula span.cm-operator { color: #ff79c6; }
.cm-s-dracula span.cm-keyword { color: #ff79c6; }
.cm-s-dracula span.cm-atom { color: #bd93f9; }
.cm-s-dracula span.cm-meta { color: #f8f8f2; }
.cm-s-dracula span.cm-tag { color: #ff79c6; }
.cm-s-dracula span.cm-attribute { color: #50fa7b; }
.cm-s-dracula span.cm-qualifier { color: #50fa7b; }
.cm-s-dracula span.cm-property { color: #66d9ef; }
.cm-s-dracula span.cm-builtin { color: #50fa7b; }
.cm-s-dracula span.cm-variable-3, .cm-s-dracula span.cm-type { color: #ffb86c; }

.cm-s-dracula .CodeMirror-activeline-background { background: rgba(255,255,255,0.1); }
.cm-s-dracula .CodeMirror-matchingbracket { text-decoration: underline; color: white !important; }

/*

    Name:       Hopscotch
    Author:     Jan T. Sott

    CodeMirror template by Jan T. Sott (https://github.com/idleberg/base16-codemirror)
    Original Base16 color scheme by Chris Kempson (https://github.com/chriskempson/base16)

*/

.cm-s-hopscotch.CodeMirror {background: #322931; color: #d5d3d5;}
.cm-s-hopscotch div.CodeMirror-selected {background: #433b42 !important;}
.cm-s-hopscotch .CodeMirror-gutters {background: #322931; border-right: 0px;}
.cm-s-hopscotch .CodeMirror-linenumber {color: #797379;}
.cm-s-hopscotch .CodeMirror-cursor {border-left: 1px solid #989498 !important;}

.cm-s-hopscotch span.cm-comment {color: #b33508;}
.cm-s-hopscotch span.cm-atom {color: #c85e7c;}
.cm-s-hopscotch span.cm-number {color: #c85e7c;}

.cm-s-hopscotch span.cm-property, .cm-s-hopscotch span.cm-attribute {color: #8fc13e;}
.cm-s-hopscotch span.cm-keyword {color: #dd464c;}
.cm-s-hopscotch span.cm-string {color: #fdcc59;}

.cm-s-hopscotch span.cm-variable {color: #8fc13e;}
.cm-s-hopscotch span.cm-variable-2 {color: #1290bf;}
.cm-s-hopscotch span.cm-def {color: #fd8b19;}
.cm-s-hopscotch span.cm-error {background: #dd464c; color: #989498;}
.cm-s-hopscotch span.cm-bracket {color: #d5d3d5;}
.cm-s-hopscotch span.cm-tag {color: #dd464c;}
.cm-s-hopscotch span.cm-link {color: #c85e7c;}

.cm-s-hopscotch .CodeMirror-matchingbracket { text-decoration: underline; color: white !important;}
.cm-s-hopscotch .CodeMirror-activeline-background { background: #302020; }

/****************************************************************/
/*   Based on mbonaci's Brackets mbo theme                      */
/*   https://github.com/mbonaci/global/blob/master/Mbo.tmTheme  */
/*   Create your own: http://tmtheme-editor.herokuapp.com       */
/****************************************************************/

.cm-s-mbo.CodeMirror { background: #2c2c2c; color: #ffffec; }
.cm-s-mbo div.CodeMirror-selected { background: #716C62; }
.cm-s-mbo .CodeMirror-line::selection, .cm-s-mbo .CodeMirror-line > span::selection, .cm-s-mbo .CodeMirror-line > span > span::selection { background: rgba(113, 108, 98, .99); }
.cm-s-mbo .CodeMirror-line::-moz-selection, .cm-s-mbo .CodeMirror-line > span::-moz-selection, .cm-s-mbo .CodeMirror-line > span > span::-moz-selection { background: rgba(113, 108, 98, .99); }
.cm-s-mbo .CodeMirror-gutters { background: #4e4e4e; border-right: 0px; }
.cm-s-mbo .CodeMirror-guttermarker { color: white; }
.cm-s-mbo .CodeMirror-guttermarker-subtle { color: grey; }
.cm-s-mbo .CodeMirror-linenumber { color: #dadada; }
.cm-s-mbo .CodeMirror-cursor { border-left: 1px solid #ffffec; }

.cm-s-mbo span.cm-comment { color: #95958a; }
.cm-s-mbo span.cm-atom { color: #00a8c6; }
.cm-s-mbo span.cm-number { color: #00a8c6; }

.cm-s-mbo span.cm-property, .cm-s-mbo span.cm-attribute { color: #9ddfe9; }
.cm-s-mbo span.cm-keyword { color: #ffb928; }
.cm-s-mbo span.cm-string { color: #ffcf6c; }
.cm-s-mbo span.cm-string.cm-property { color: #ffffec; }

.cm-s-mbo span.cm-variable { color: #ffffec; }
.cm-s-mbo span.cm-variable-2 { color: #00a8c6; }
.cm-s-mbo span.cm-def { color: #ffffec; }
.cm-s-mbo span.cm-bracket { color: #fffffc; font-weight: bold; }
.cm-s-mbo span.cm-tag { color: #9ddfe9; }
.cm-s-mbo span.cm-link { color: #f54b07; }
.cm-s-mbo span.cm-error { border-bottom: #636363; color: #ffffec; }
.cm-s-mbo span.cm-qualifier { color: #ffffec; }

.cm-s-mbo .CodeMirror-activeline-background { background: #494b41; }
.cm-s-mbo .CodeMirror-matchingbracket { color: #ffb928 !important; }
.cm-s-mbo .CodeMirror-matchingtag { background: rgba(255, 255, 255, .37); }

/*
  MDN-LIKE Theme - Mozilla
  Ported to CodeMirror by Peter Kroon <plakroon@gmail.com>
  Report bugs/issues here: https://github.com/codemirror/CodeMirror/issues
  GitHub: @peterkroon

  The mdn-like theme is inspired on the displayed code examples at: https://developer.mozilla.org/en-US/docs/Web/CSS/animation

*/
.cm-s-mdn-like.CodeMirror { color: #999; background-color: #fff; }
.cm-s-mdn-like div.CodeMirror-selected { background: #cfc; }
.cm-s-mdn-like .CodeMirror-line::selection, .cm-s-mdn-like .CodeMirror-line > span::selection, .cm-s-mdn-like .CodeMirror-line > span > span::selection { background: #cfc; }
.cm-s-mdn-like .CodeMirror-line::-moz-selection, .cm-s-mdn-like .CodeMirror-line > span::-moz-selection, .cm-s-mdn-like .CodeMirror-line > span > span::-moz-selection { background: #cfc; }

.cm-s-mdn-like .CodeMirror-gutters { background: #f8f8f8; border-left: 6px solid rgba(0,83,159,0.65); color: #333; }
.cm-s-mdn-like .CodeMirror-linenumber { color: #aaa; padding-left: 8px; }
.cm-s-mdn-like .CodeMirror-cursor { border-left: 2px solid #222; }

.cm-s-mdn-like .cm-keyword { color: #6262FF; }
.cm-s-mdn-like .cm-atom { color: #F90; }
.cm-s-mdn-like .cm-number { color:  #ca7841; }
.cm-s-mdn-like .cm-def { color: #8DA6CE; }
.cm-s-mdn-like span.cm-variable-2, .cm-s-mdn-like span.cm-tag { color: #690; }
.cm-s-mdn-like span.cm-variable-3, .cm-s-mdn-like span.cm-def, .cm-s-mdn-like span.cm-type { color: #07a; }

.cm-s-mdn-like .cm-variable { color: #07a; }
.cm-s-mdn-like .cm-property { color: #905; }
.cm-s-mdn-like .cm-qualifier { color: #690; }

.cm-s-mdn-like .cm-operator { color: #cda869; }
.cm-s-mdn-like .cm-comment { color:#777; font-weight:normal; }
.cm-s-mdn-like .cm-string { color:#07a; font-style:italic; }
.cm-s-mdn-like .cm-string-2 { color:#bd6b18; } /*?*/
.cm-s-mdn-like .cm-meta { color: #000; } /*?*/
.cm-s-mdn-like .cm-builtin { color: #9B7536; } /*?*/
.cm-s-mdn-like .cm-tag { color: #997643; }
.cm-s-mdn-like .cm-attribute { color: #d6bb6d; } /*?*/
.cm-s-mdn-like .cm-header { color: #FF6400; }
.cm-s-mdn-like .cm-hr { color: #AEAEAE; }
.cm-s-mdn-like .cm-link { color:#ad9361; font-style:italic; text-decoration:none; }
.cm-s-mdn-like .cm-error { border-bottom: 1px solid red; }

div.cm-s-mdn-like .CodeMirror-activeline-background { background: #efefff; }
div.cm-s-mdn-like span.CodeMirror-matchingbracket { outline:1px solid grey; color: inherit; }

.cm-s-mdn-like.CodeMirror { background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFcAAAAyCAYAAAAp8UeFAAAHvklEQVR42s2b63bcNgyEQZCSHCdt2vd/0tWF7I+Q6XgMXiTtuvU5Pl57ZQKkKHzEAOtF5KeIJBGJ8uvL599FRFREZhFx8DeXv8trn68RuGaC8TRfo3SNp9dlDDHedyLyTUTeRWStXKPZrjtpZxaRw5hPqozRs1N8/enzIiQRWcCgy4MUA0f+XWliDhyL8Lfyvx7ei/Ae3iQFHyw7U/59pQVIMEEPEz0G7XiwdRjzSfC3UTtz9vchIntxvry5iMgfIhJoEflOz2CQr3F5h/HfeFe+GTdLaKcu9L8LTeQb/R/7GgbsfKedyNdoHsN31uRPWrfZ5wsj/NzzRQHuToIdU3ahwnsKPxXCjJITuOsi7XLc7SG/v5GdALs7wf8JjTFiB5+QvTEfRyGOfX3Lrx8wxyQi3sNq46O7QahQiCsRFgqddjBouVEHOKDgXAQHD9gJCr5sMKkEdjwsarG/ww3BMHBU7OBjXnzdyY7SfCxf5/z6ATccrwlKuwC/jhznnPF4CgVzhhVf4xp2EixcBActO75iZ8/fM9zAs2OMzKdslgXWJ9XG8PQoOAMA5fGcsvORgv0doBXyHrCwfLJAOwo71QLNkb8n2Pl6EWiR7OCibtkPaz4Kc/0NNAze2gju3zOwekALDaCFPI5vjPFmgGY5AZqyGEvH1x7QfIb8YtxMnA/b+QQ0aQDAwc6JMFg8CbQZ4qoYEEHbRwNojuK3EHwd7VALSgq+MNDKzfT58T8qdpADrgW0GmgcAS1lhzztJmkAzcPNOQbsWEALBDSlMKUG0Eq4CLAQWvEVQ9WU57gZJwZtgPO3r9oBTQ9WO8TjqXINx8R0EYpiZEUWOF3FxkbJkgU9B2f41YBrIj5ZfsQa0M5kTgiAAqM3ShXLgu8XMqcrQBvJ0CL5pnTsfMB13oB8athpAq2XOQmcGmoACCLydx7nToa23ATaSIY2ichfOdPTGxlasXMLaL0MLZAOwAKIM+y8CmicobGdCcbbK9DzN+yYGVoNNI5iUKTMyYOjPse4A8SM1MmcXgU0toOq1yO/v8FOxlASyc7TgeYaAMBJHcY1CcCwGI/TK4AmDbDyKYBBtFUkRwto8gygiQEaByFgJ00BH2M8JWwQS1nafDXQCidWyOI8AcjDCSjCLk8ngObuAm3JAHAdubAmOaK06V8MNEsKPJOhobSprwQa6gD7DclRQdqcwL4zxqgBrQcabUiBLclRDKAlWp+etPkBaNMA0AKlrHwTdEByZAA4GM+SNluSY6wAzcMNewxmgig5Ks0nkrSpBvSaQHMdKTBAnLojOdYyGpQ254602ZILPdTD1hdlggdIm74jbTp8vDwF5ZYUeLWGJpWsh6XNyXgcYwVoJQTEhhTYkxzZjiU5npU2TaB979TQehlaAVq4kaGpiPwwwLkYUuBbQwocyQTv1tA0+1UFWoJF3iv1oq+qoSk8EQdJmwHkziIF7oOZk14EGitibAdjLYYK78H5vZOhtWpoI0ATGHs0Q8OMb4Ey+2bU2UYztCtA0wFAs7TplGLRVQCcqaFdGSPCeTI1QNIC52iWNzof6Uib7xjEp07mNNoUYmVosVItHrHzRlLgBn9LFyRHaQCtVUMbtTNhoXWiTOO9k/V8BdAc1Oq0ArSQs6/5SU0hckNy9NnXqQY0PGYo5dWJ7nINaN6o958FWin27aBaWRka1r5myvLOAm0j30eBJqCxHLReVclxhxOEN2JfDWjxBtAC7MIH1fVaGdoOp4qJYDgKtKPSFNID2gSnGldrCqkFZ+5UeQXQBIRrSwocbdZYQT/2LwRahBPBXoHrB8nxaGROST62DKUbQOMMzZIC9abkuELfQzQALWTnDNAm8KHWFOJgJ5+SHIvTPcmx1xQyZRhNL5Qci689aXMEaN/uNIWkEwDAvFpOZmgsBaaGnbs1NPa1Jm32gBZAIh1pCtG7TSH4aE0y1uVY4uqoFPisGlpP2rSA5qTecWn5agK6BzSpgAyD+wFaqhnYoSZ1Vwr8CmlTQbrcO3ZaX0NAEyMbYaAlyquFoLKK3SPby9CeVUPThrSJmkCAE0CrKUQadi4DrdSlWhmah0YL9z9vClH59YGbHx1J8VZTyAjQepJjmXwAKTDQI3omc3p1U4gDUf6RfcdYfrUp5ClAi2J3Ba6UOXGo+K+bQrjjssitG2SJzshaLwMtXgRagUNpYYoVkMSBLM+9GGiJZMvduG6DRZ4qc04DMPtQQxOjEtACmhO7K1AbNbQDEggZyJwscFpAGwENhoBeUwh3bWolhe8BTYVKxQEWrSUn/uhcM5KhvUu/+eQu0Lzhi+VrK0PrZZNDQKs9cpYUuFYgMVpD4/NxenJTiMCNqdUEUf1qZWjppLT5qSkkUZbCwkbZMSuVnu80hfSkzRbQeqCZSAh6huR4VtoM2gHAlLf72smuWgE+VV7XpE25Ab2WFDgyhnSuKbs4GuGzCjR+tIoUuMFg3kgcWKLTwRqanJQ2W00hAsenfaApRC42hbCvK1SlE0HtE9BGgneJO+ELamitD1YjjOYnNYVcraGhtKkW0EqVVeDx733I2NH581k1NNxNLG0i0IJ8/NjVaOZ0tYZ2Vtr0Xv7tPV3hkWp9EFkgS/J0vosngTaSoaG06WHi+xObQkaAdlbanP8B2+2l0f90LmUAAAAASUVORK5CYII=); }

/*

    Name:       seti
    Author:     Michael Kaminsky (http://github.com/mkaminsky11)

    Original seti color scheme by Jesse Weed (https://github.com/jesseweed/seti-syntax)

*/


.cm-s-seti.CodeMirror {
  background-color: #151718 !important;
  color: #CFD2D1 !important;
  border: none;
}
.cm-s-seti .CodeMirror-gutters {
  color: #404b53;
  background-color: #0E1112;
  border: none;
}
.cm-s-seti .CodeMirror-cursor { border-left: solid thin #f8f8f0; }
.cm-s-seti .CodeMirror-linenumber { color: #6D8A88; }
.cm-s-seti.CodeMirror-focused div.CodeMirror-selected { background: rgba(255, 255, 255, 0.10); }
.cm-s-seti .CodeMirror-line::selection, .cm-s-seti .CodeMirror-line > span::selection, .cm-s-seti .CodeMirror-line > span > span::selection { background: rgba(255, 255, 255, 0.10); }
.cm-s-seti .CodeMirror-line::-moz-selection, .cm-s-seti .CodeMirror-line > span::-moz-selection, .cm-s-seti .CodeMirror-line > span > span::-moz-selection { background: rgba(255, 255, 255, 0.10); }
.cm-s-seti span.cm-comment { color: #41535b; }
.cm-s-seti span.cm-string, .cm-s-seti span.cm-string-2 { color: #55b5db; }
.cm-s-seti span.cm-number { color: #cd3f45; }
.cm-s-seti span.cm-variable { color: #55b5db; }
.cm-s-seti span.cm-variable-2 { color: #a074c4; }
.cm-s-seti span.cm-def { color: #55b5db; }
.cm-s-seti span.cm-keyword { color: #ff79c6; }
.cm-s-seti span.cm-operator { color: #9fca56; }
.cm-s-seti span.cm-keyword { color: #e6cd69; }
.cm-s-seti span.cm-atom { color: #cd3f45; }
.cm-s-seti span.cm-meta { color: #55b5db; }
.cm-s-seti span.cm-tag { color: #55b5db; }
.cm-s-seti span.cm-attribute { color: #9fca56; }
.cm-s-seti span.cm-qualifier { color: #9fca56; }
.cm-s-seti span.cm-property { color: #a074c4; }
.cm-s-seti span.cm-variable-3, .cm-s-seti span.cm-type { color: #9fca56; }
.cm-s-seti span.cm-builtin { color: #9fca56; }
.cm-s-seti .CodeMirror-activeline-background { background: #101213; }
.cm-s-seti .CodeMirror-matchingbracket { text-decoration: underline; color: white !important; }

/*
Solarized theme for code-mirror
http://ethanschoonover.com/solarized
*/

/*
Solarized color palette
http://ethanschoonover.com/solarized/img/solarized-palette.png
*/

.solarized.base03 { color: #002b36; }
.solarized.base02 { color: #073642; }
.solarized.base01 { color: #586e75; }
.solarized.base00 { color: #657b83; }
.solarized.base0 { color: #839496; }
.solarized.base1 { color: #93a1a1; }
.solarized.base2 { color: #eee8d5; }
.solarized.base3  { color: #fdf6e3; }
.solarized.solar-yellow  { color: #b58900; }
.solarized.solar-orange  { color: #cb4b16; }
.solarized.solar-red { color: #dc322f; }
.solarized.solar-magenta { color: #d33682; }
.solarized.solar-violet  { color: #6c71c4; }
.solarized.solar-blue { color: #268bd2; }
.solarized.solar-cyan { color: #2aa198; }
.solarized.solar-green { color: #859900; }

/* Color scheme for code-mirror */

.cm-s-solarized {
  line-height: 1.45em;
  color-profile: sRGB;
  rendering-intent: auto;
}
.cm-s-solarized.cm-s-dark {
  color: #839496;
  background-color: #002b36;
  text-shadow: #002b36 0 1px;
}
.cm-s-solarized.cm-s-light {
  background-color: #fdf6e3;
  color: #657b83;
  text-shadow: #eee8d5 0 1px;
}

.cm-s-solarized .CodeMirror-widget {
  text-shadow: none;
}

.cm-s-solarized .cm-header { color: #586e75; }
.cm-s-solarized .cm-quote { color: #93a1a1; }

.cm-s-solarized .cm-keyword { color: #cb4b16; }
.cm-s-solarized .cm-atom { color: #d33682; }
.cm-s-solarized .cm-number { color: #d33682; }
.cm-s-solarized .cm-def { color: #2aa198; }

.cm-s-solarized .cm-variable { color: #839496; }
.cm-s-solarized .cm-variable-2 { color: #b58900; }
.cm-s-solarized .cm-variable-3, .cm-s-solarized .cm-type { color: #6c71c4; }

.cm-s-solarized .cm-property { color: #2aa198; }
.cm-s-solarized .cm-operator { color: #6c71c4; }

.cm-s-solarized .cm-comment { color: #586e75; font-style:italic; }

.cm-s-solarized .cm-string { color: #859900; }
.cm-s-solarized .cm-string-2 { color: #b58900; }

.cm-s-solarized .cm-meta { color: #859900; }
.cm-s-solarized .cm-qualifier { color: #b58900; }
.cm-s-solarized .cm-builtin { color: #d33682; }
.cm-s-solarized .cm-bracket { color: #cb4b16; }
.cm-s-solarized .CodeMirror-matchingbracket { color: #859900; }
.cm-s-solarized .CodeMirror-nonmatchingbracket { color: #dc322f; }
.cm-s-solarized .cm-tag { color: #93a1a1; }
.cm-s-solarized .cm-attribute { color: #2aa198; }
.cm-s-solarized .cm-hr {
  color: transparent;
  border-top: 1px solid #586e75;
  display: block;
}
.cm-s-solarized .cm-link { color: #93a1a1; cursor: pointer; }
.cm-s-solarized .cm-special { color: #6c71c4; }
.cm-s-solarized .cm-em {
  color: #999;
  text-decoration: underline;
  text-decoration-style: dotted;
}
.cm-s-solarized .cm-error,
.cm-s-solarized .cm-invalidchar {
  color: #586e75;
  border-bottom: 1px dotted #dc322f;
}

.cm-s-solarized.cm-s-dark div.CodeMirror-selected { background: #073642; }
.cm-s-solarized.cm-s-dark.CodeMirror ::selection { background: rgba(7, 54, 66, 0.99); }
.cm-s-solarized.cm-s-dark .CodeMirror-line::-moz-selection, .cm-s-dark .CodeMirror-line > span::-moz-selection, .cm-s-dark .CodeMirror-line > span > span::-moz-selection { background: rgba(7, 54, 66, 0.99); }

.cm-s-solarized.cm-s-light div.CodeMirror-selected { background: #eee8d5; }
.cm-s-solarized.cm-s-light .CodeMirror-line::selection, .cm-s-light .CodeMirror-line > span::selection, .cm-s-light .CodeMirror-line > span > span::selection { background: #eee8d5; }
.cm-s-solarized.cm-s-light .CodeMirror-line::-moz-selection, .cm-s-ligh .CodeMirror-line > span::-moz-selection, .cm-s-ligh .CodeMirror-line > span > span::-moz-selection { background: #eee8d5; }

/* Editor styling */



/* Little shadow on the view-port of the buffer view */
.cm-s-solarized.CodeMirror {
  -moz-box-shadow: inset 7px 0 12px -6px #000;
  -webkit-box-shadow: inset 7px 0 12px -6px #000;
  box-shadow: inset 7px 0 12px -6px #000;
}

/* Remove gutter border */
.cm-s-solarized .CodeMirror-gutters {
  border-right: 0;
}

/* Gutter colors and line number styling based of color scheme (dark / light) */

/* Dark */
.cm-s-solarized.cm-s-dark .CodeMirror-gutters {
  background-color: #073642;
}

.cm-s-solarized.cm-s-dark .CodeMirror-linenumber {
  color: #586e75;
  text-shadow: #021014 0 -1px;
}

/* Light */
.cm-s-solarized.cm-s-light .CodeMirror-gutters {
  background-color: #eee8d5;
}

.cm-s-solarized.cm-s-light .CodeMirror-linenumber {
  color: #839496;
}

/* Common */
.cm-s-solarized .CodeMirror-linenumber {
  padding: 0 5px;
}
.cm-s-solarized .CodeMirror-guttermarker-subtle { color: #586e75; }
.cm-s-solarized.cm-s-dark .CodeMirror-guttermarker { color: #ddd; }
.cm-s-solarized.cm-s-light .CodeMirror-guttermarker { color: #cb4b16; }

.cm-s-solarized .CodeMirror-gutter .CodeMirror-gutter-text {
  color: #586e75;
}

/* Cursor */
.cm-s-solarized .CodeMirror-cursor { border-left: 1px solid #819090; }

/* Fat cursor */
.cm-s-solarized.cm-s-light.cm-fat-cursor .CodeMirror-cursor { background: #77ee77; }
.cm-s-solarized.cm-s-light .cm-animate-fat-cursor { background-color: #77ee77; }
.cm-s-solarized.cm-s-dark.cm-fat-cursor .CodeMirror-cursor { background: #586e75; }
.cm-s-solarized.cm-s-dark .cm-animate-fat-cursor { background-color: #586e75; }

/* Active line */
.cm-s-solarized.cm-s-dark .CodeMirror-activeline-background {
  background: rgba(255, 255, 255, 0.06);
}
.cm-s-solarized.cm-s-light .CodeMirror-activeline-background {
  background: rgba(0, 0, 0, 0.06);
}

.cm-s-the-matrix.CodeMirror { background: #000000; color: #00FF00; }
.cm-s-the-matrix div.CodeMirror-selected { background: #2D2D2D; }
.cm-s-the-matrix .CodeMirror-line::selection, .cm-s-the-matrix .CodeMirror-line > span::selection, .cm-s-the-matrix .CodeMirror-line > span > span::selection { background: rgba(45, 45, 45, 0.99); }
.cm-s-the-matrix .CodeMirror-line::-moz-selection, .cm-s-the-matrix .CodeMirror-line > span::-moz-selection, .cm-s-the-matrix .CodeMirror-line > span > span::-moz-selection { background: rgba(45, 45, 45, 0.99); }
.cm-s-the-matrix .CodeMirror-gutters { background: #060; border-right: 2px solid #00FF00; }
.cm-s-the-matrix .CodeMirror-guttermarker { color: #0f0; }
.cm-s-the-matrix .CodeMirror-guttermarker-subtle { color: white; }
.cm-s-the-matrix .CodeMirror-linenumber { color: #FFFFFF; }
.cm-s-the-matrix .CodeMirror-cursor { border-left: 1px solid #00FF00; }

.cm-s-the-matrix span.cm-keyword { color: #008803; font-weight: bold; }
.cm-s-the-matrix span.cm-atom { color: #3FF; }
.cm-s-the-matrix span.cm-number { color: #FFB94F; }
.cm-s-the-matrix span.cm-def { color: #99C; }
.cm-s-the-matrix span.cm-variable { color: #F6C; }
.cm-s-the-matrix span.cm-variable-2 { color: #C6F; }
.cm-s-the-matrix span.cm-variable-3, .cm-s-the-matrix span.cm-type { color: #96F; }
.cm-s-the-matrix span.cm-property { color: #62FFA0; }
.cm-s-the-matrix span.cm-operator { color: #999; }
.cm-s-the-matrix span.cm-comment { color: #CCCCCC; }
.cm-s-the-matrix span.cm-string { color: #39C; }
.cm-s-the-matrix span.cm-meta { color: #C9F; }
.cm-s-the-matrix span.cm-qualifier { color: #FFF700; }
.cm-s-the-matrix span.cm-builtin { color: #30a; }
.cm-s-the-matrix span.cm-bracket { color: #cc7; }
.cm-s-the-matrix span.cm-tag { color: #FFBD40; }
.cm-s-the-matrix span.cm-attribute { color: #FFF700; }
.cm-s-the-matrix span.cm-error { color: #FF0000; }

.cm-s-the-matrix .CodeMirror-activeline-background { background: #040; }

/*
Copyright (C) 2011 by MarkLogic Corporation
Author: Mike Brevoort <mike@brevoort.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
.cm-s-xq-light span.cm-keyword { line-height: 1em; font-weight: bold; color: #5A5CAD; }
.cm-s-xq-light span.cm-atom { color: #6C8CD5; }
.cm-s-xq-light span.cm-number { color: #164; }
.cm-s-xq-light span.cm-def { text-decoration:underline; }
.cm-s-xq-light span.cm-variable { color: black; }
.cm-s-xq-light span.cm-variable-2 { color:black; }
.cm-s-xq-light span.cm-variable-3, .cm-s-xq-light span.cm-type { color: black; }
.cm-s-xq-light span.cm-property {}
.cm-s-xq-light span.cm-operator {}
.cm-s-xq-light span.cm-comment { color: #0080FF; font-style: italic; }
.cm-s-xq-light span.cm-string { color: red; }
.cm-s-xq-light span.cm-meta { color: yellow; }
.cm-s-xq-light span.cm-qualifier { color: grey; }
.cm-s-xq-light span.cm-builtin { color: #7EA656; }
.cm-s-xq-light span.cm-bracket { color: #cc7; }
.cm-s-xq-light span.cm-tag { color: #3F7F7F; }
.cm-s-xq-light span.cm-attribute { color: #7F007F; }
.cm-s-xq-light span.cm-error { color: #f00; }

.cm-s-xq-light .CodeMirror-activeline-background { background: #e8f2ff; }
.cm-s-xq-light .CodeMirror-matchingbracket { outline:1px solid grey;color:black !important;background:yellow; }

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.CodeMirror {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;
  /* Changed to auto to autogrow */
}

.CodeMirror pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* This causes https://github.com/jupyter/jupyterlab/issues/522 */
/* May not cause it not because we changed it! */
.CodeMirror-lines {
  padding: var(--jp-code-padding) 0;
}

.CodeMirror-linenumber {
  padding: 0 8px;
}

.jp-CodeMirrorEditor-static {
  margin: var(--jp-code-padding);
}

.jp-CodeMirrorEditor,
.jp-CodeMirrorEditor-static {
  cursor: text;
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.CodeMirror.jp-mod-readOnly .CodeMirror-cursor {
  display: none;
}

.CodeMirror-gutters {
  border-right: 1px solid var(--jp-border-color2);
  background-color: var(--jp-layout-color0);
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.CodeMirror-selectedtext.cm-searching {
  background-color: var(--jp-search-selected-match-background-color) !important;
  color: var(--jp-search-selected-match-color) !important;
}

.cm-searching {
  background-color: var(
    --jp-search-unselected-match-background-color
  ) !important;
  color: var(--jp-search-unselected-match-color) !important;
}

.CodeMirror-focused .CodeMirror-selected {
  background-color: var(--jp-editor-selected-focused-background);
}

.CodeMirror-selected {
  background-color: var(--jp-editor-selected-background);
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/**
 * Here is our jupyter theme for CodeMirror syntax highlighting
 * This is used in our marked.js syntax highlighting and CodeMirror itself
 * The string "jupyter" is set in ../codemirror/widget.DEFAULT_CODEMIRROR_THEME
 * This came from the classic notebook, which came form highlight.js/GitHub
 */

/**
 * CodeMirror themes are handling the background/color in this way. This works
 * fine for CodeMirror editors outside the notebook, but the notebook styles
 * these things differently.
 */
.CodeMirror.cm-s-jupyter {
  background: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* In the notebook, we want this styling to be handled by its container */
.jp-CodeConsole .CodeMirror.cm-s-jupyter,
.jp-Notebook .CodeMirror.cm-s-jupyter {
  background: transparent;
}

.cm-s-jupyter .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}
.cm-s-jupyter span.cm-keyword {
  color: var(--jp-mirror-editor-keyword-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-atom {
  color: var(--jp-mirror-editor-atom-color);
}
.cm-s-jupyter span.cm-number {
  color: var(--jp-mirror-editor-number-color);
}
.cm-s-jupyter span.cm-def {
  color: var(--jp-mirror-editor-def-color);
}
.cm-s-jupyter span.cm-variable {
  color: var(--jp-mirror-editor-variable-color);
}
.cm-s-jupyter span.cm-variable-2 {
  color: var(--jp-mirror-editor-variable-2-color);
}
.cm-s-jupyter span.cm-variable-3 {
  color: var(--jp-mirror-editor-variable-3-color);
}
.cm-s-jupyter span.cm-punctuation {
  color: var(--jp-mirror-editor-punctuation-color);
}
.cm-s-jupyter span.cm-property {
  color: var(--jp-mirror-editor-property-color);
}
.cm-s-jupyter span.cm-operator {
  color: var(--jp-mirror-editor-operator-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-comment {
  color: var(--jp-mirror-editor-comment-color);
  font-style: italic;
}
.cm-s-jupyter span.cm-string {
  color: var(--jp-mirror-editor-string-color);
}
.cm-s-jupyter span.cm-string-2 {
  color: var(--jp-mirror-editor-string-2-color);
}
.cm-s-jupyter span.cm-meta {
  color: var(--jp-mirror-editor-meta-color);
}
.cm-s-jupyter span.cm-qualifier {
  color: var(--jp-mirror-editor-qualifier-color);
}
.cm-s-jupyter span.cm-builtin {
  color: var(--jp-mirror-editor-builtin-color);
}
.cm-s-jupyter span.cm-bracket {
  color: var(--jp-mirror-editor-bracket-color);
}
.cm-s-jupyter span.cm-tag {
  color: var(--jp-mirror-editor-tag-color);
}
.cm-s-jupyter span.cm-attribute {
  color: var(--jp-mirror-editor-attribute-color);
}
.cm-s-jupyter span.cm-header {
  color: var(--jp-mirror-editor-header-color);
}
.cm-s-jupyter span.cm-quote {
  color: var(--jp-mirror-editor-quote-color);
}
.cm-s-jupyter span.cm-link {
  color: var(--jp-mirror-editor-link-color);
}
.cm-s-jupyter span.cm-error {
  color: var(--jp-mirror-editor-error-color);
}
.cm-s-jupyter span.cm-hr {
  color: #999;
}

.cm-s-jupyter span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}

.cm-s-jupyter .CodeMirror-activeline-background,
.cm-s-jupyter .CodeMirror-gutter {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0px;
  padding: 0px;
  line-height: normal;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}
.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}
.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}
.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}
.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}
.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}
.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}
.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
}
.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
}
.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
}
.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
}
.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
}
.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
}
.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
}
.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}
.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}
.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}
.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}
.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}
.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}
.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}
.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
}
.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
}
.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
}
.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
}
.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
}
.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
}
.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
}
.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}
.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}
.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0em;
}

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: 12px;
  table-layout: fixed;
  margin-left: auto;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon table {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0px;
}

.jp-RenderedHTMLCommon p {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}
/* ...or leave it untouched if they don't */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-dark-background {
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-light-background {
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}
.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}
.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}
.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}
.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}
.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}
.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}
.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}
.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: 0.8em;
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  border-bottom: none;
  height: auto;
  margin: var(--jp-toolbar-header-margin);
  box-shadow: none;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 4px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0px 2px;
  padding: 0px 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0px;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar.jp-Toolbar {
  padding: 0px;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  justify-content: space-evenly;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-Toolbar-item {
  flex: 1;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-ToolbarButtonComponent {
  width: 100%;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px 12px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-item.jp-mod-selected {
  color: white;
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon:before {
  color: limegreen;
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0px;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-DirListing-deadSpace {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

.jp-FileDialog.jp-mod-conflict input {
  color: red;
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
}

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: flex;
  flex-direction: row;
}

.jp-OutputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-output {
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea-child .jp-OutputArea-output {
  flex-grow: 1;
  flex-shrink: 1;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0px;
  padding: 0px;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0px;
  flex: 1 1 auto;
}

.jp-OutputArea-executeResult.jp-RenderedText {
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-OutputArea-stdin {
  line-height: var(--jp-code-line-height);
  padding-top: var(--jp-code-padding);
  display: flex;
}

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;
  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0px;
  bottom: 0px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0px;
  width: 100%;
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: flex;
  flex-direction: row;
}

.jp-InputArea-editor {
  flex: 1 1 auto;
}

.jp-InputArea-editor {
  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0px;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: flex;
  flex-direction: row;
  flex: 1 1 auto;
}

.jp-Placeholder-prompt {
  box-sizing: border-box;
}

.jp-Placeholder-content {
  flex: 1 1 auto;
  border: none;
  background: transparent;
  height: 20px;
  box-sizing: border-box;
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0px;
  margin: 0px;
  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 200px;
  box-shadow: inset 0 0 6px 2px rgba(0, 0, 0, 0.3);
  margin-left: var(--jp-private-cell-scrolling-output-offset);
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  flex: 0 0
    calc(
      var(--jp-cell-prompt-width) -
        var(--jp-private-cell-scrolling-output-offset)
    );
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  flex: 1 1 auto;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: 2px;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: flex;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0px;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-NotebookTools-tool {
  padding: 0px 12px 0 12px;
}

.jp-ActiveCellTool {
  padding: 12px;
  background-color: var(--jp-layout-color1);
  border-top: none !important;
}

.jp-ActiveCellTool .jp-InputArea-prompt {
  flex: 0 0 auto;
  padding-left: 0px;
}

.jp-ActiveCellTool .jp-InputArea-editor {
  flex: 1 1 auto;
  background: var(--jp-cell-editor-background);
  border-color: var(--jp-cell-editor-border-color);
}

.jp-ActiveCellTool .jp-InputArea-editor .CodeMirror {
  background: transparent;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0px 12px 0px;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label {
  line-height: 1.4;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensurePackage() in @jupyterlab/buildutils */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

</style>

    <style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0px 2px 1px -1px var(--jp-shadow-umbra-color),
    0px 1px 1px 0px var(--jp-shadow-penumbra-color),
    0px 1px 3px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0px 3px 1px -2px var(--jp-shadow-umbra-color),
    0px 2px 2px 0px var(--jp-shadow-penumbra-color),
    0px 1px 5px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0px 2px 4px -1px var(--jp-shadow-umbra-color),
    0px 4px 5px 0px var(--jp-shadow-penumbra-color),
    0px 1px 10px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0px 3px 5px -1px var(--jp-shadow-umbra-color),
    0px 6px 10px 0px var(--jp-shadow-penumbra-color),
    0px 1px 18px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0px 5px 5px -3px var(--jp-shadow-umbra-color),
    0px 8px 10px 1px var(--jp-shadow-penumbra-color),
    0px 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0px 7px 8px -4px var(--jp-shadow-umbra-color),
    0px 12px 17px 2px var(--jp-shadow-penumbra-color),
    0px 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0px 8px 10px -5px var(--jp-shadow-umbra-color),
    0px 16px 24px 2px var(--jp-shadow-penumbra-color),
    0px 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0px 10px 13px -6px var(--jp-shadow-umbra-color),
    0px 20px 31px 3px var(--jp-shadow-penumbra-color),
    0px 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0px 11px 15px -7px var(--jp-shadow-umbra-color),
    0px 24px 38px 3px var(--jp-shadow-penumbra-color),
    0px 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;

  --jp-ui-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica,
    Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;

  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);

  --jp-content-link-color: var(--md-blue-700);

  --jp-content-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: Menlo, Consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-700);
  --jp-brand-color1: var(--md-blue-500);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);

  --jp-accent-color0: var(--md-green-700);
  --jp-accent-color1: var(--md-green-500);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-700);
  --jp-warn-color1: var(--md-orange-500);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-700);
  --jp-error-color1: var(--md-red-500);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-700);
  --jp-success-color1: var(--md-green-500);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-700);
  --jp-info-color1: var(--md-cyan-500);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;

  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;

  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);

  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: 'Source Code Pro', monospace;
  --jp-cell-prompt-letter-spacing: 0px;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);
  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;
  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0px 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-border-color1);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: #05a;
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #aa22ff;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #aa22ff;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 180px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);
}
</style>

<style type="text/css">
a.anchor-link {
   display: none;
}
.highlight  {
    margin: 0.4em;
}

/* Input area styling */
.jp-InputArea {
    overflow: hidden;
}

.jp-InputArea-editor {
    overflow: hidden;
}

@media print {
  body {
    margin: 0;
  }
}
</style>



<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML-full,Safe"> </script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: { 
                    automatic: true 
                    }
                },
                "HTML-CSS": {
                    linebreaks: { 
                    automatic: true 
                    }
                }
            });
        
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
    <!-- End of mathjax configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">

<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Heart-Attack-Prediction">Heart Attack Prediction<a class="anchor-link" href="#Heart-Attack-Prediction">&#182;</a></h3><p>Rafay Mahmood</p>
<p>L1F18BSCS0095</p>
</div>
</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="Basically,-our-main-goal-is-to-predict-heart-attack-or-stroke-based-on-vitals-of-a-patient.-Our-dataset-consists-of-vitals-of-various-patients-along-with-their-general-information-like-age,-gender-etc.-Based-on-reports-and-vitals-we-will-predict-either-this-patient-had-attack-or-not">Basically, our main goal is to predict heart attack or stroke based on vitals of a patient. Our dataset consists of vitals of various patients along with their general information like age, gender etc. Based on reports and vitals we will predict either this patient had attack or not<a class="anchor-link" href="#Basically,-our-main-goal-is-to-predict-heart-attack-or-stroke-based-on-vitals-of-a-patient.-Our-dataset-consists-of-vitals-of-various-patients-along-with-their-general-information-like-age,-gender-etc.-Based-on-reports-and-vitals-we-will-predict-either-this-patient-had-attack-or-not">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[1]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">re</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span> 
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span> 
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span> 
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">string</span>
<span class="kn">import</span> <span class="nn">warnings</span> 
<span class="kn">import</span> <span class="nn">math</span>
<span class="n">warnings</span><span class="o">.</span><span class="n">filterwarnings</span><span class="p">(</span><span class="s2">&quot;ignore&quot;</span><span class="p">,</span> <span class="n">category</span><span class="o">=</span><span class="ne">DeprecationWarning</span><span class="p">)</span>

<span class="o">%</span><span class="k">matplotlib</span> inline
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[2]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#reading csv files.</span>

<span class="n">hdf</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;heart.csv&#39;</span><span class="p">)</span>
<span class="n">o2</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;o2Saturation.csv&#39;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[3]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#concating both dataset files</span>
<span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">hdf</span><span class="p">,</span> <span class="n">o2</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">join</span><span class="o">=</span><span class="s1">&#39;inner&#39;</span><span class="p">)</span>
<span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[3]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
      <th>98.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>98.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>98.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>98.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>98.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>97.5</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[4]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#renaming saturation column to o2</span>
<span class="n">df</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">{</span><span class="s1">&#39;98.6&#39;</span><span class="p">:</span><span class="s1">&#39;o2&#39;</span><span class="p">}</span> <span class="p">,</span><span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">df</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[4]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
      <th>o2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>98.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>98.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>98.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>98.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>97.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>241</td>
      <td>0</td>
      <td>1</td>
      <td>123</td>
      <td>1</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>97.5</td>
    </tr>
    <tr>
      <th>299</th>
      <td>45</td>
      <td>1</td>
      <td>3</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>132</td>
      <td>0</td>
      <td>1.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>97.5</td>
    </tr>
    <tr>
      <th>300</th>
      <td>68</td>
      <td>1</td>
      <td>0</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>1</td>
      <td>141</td>
      <td>0</td>
      <td>3.4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>97.5</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>1</td>
      <td>115</td>
      <td>1</td>
      <td>1.2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>97.5</td>
    </tr>
    <tr>
      <th>302</th>
      <td>57</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>97.5</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 15 columns</p>
</div>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[5]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">duplicated</span><span class="p">(</span><span class="n">keep</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)]</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[5]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
      <th>o2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163</th>
      <td>38</td>
      <td>1</td>
      <td>2</td>
      <td>138</td>
      <td>175</td>
      <td>0</td>
      <td>1</td>
      <td>173</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>97.5</td>
    </tr>
    <tr>
      <th>164</th>
      <td>38</td>
      <td>1</td>
      <td>2</td>
      <td>138</td>
      <td>175</td>
      <td>0</td>
      <td>1</td>
      <td>173</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>97.5</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[6]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">drop_duplicates</span><span class="p">(</span><span class="n">keep</span> <span class="o">=</span> <span class="s1">&#39;first&#39;</span><span class="p">,</span> <span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[7]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[7]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
      <th>o2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>302.00000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54.42053</td>
      <td>0.682119</td>
      <td>0.963576</td>
      <td>131.602649</td>
      <td>246.500000</td>
      <td>0.149007</td>
      <td>0.526490</td>
      <td>149.569536</td>
      <td>0.327815</td>
      <td>1.043046</td>
      <td>1.397351</td>
      <td>0.718543</td>
      <td>2.314570</td>
      <td>0.543046</td>
      <td>97.480795</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.04797</td>
      <td>0.466426</td>
      <td>1.032044</td>
      <td>17.563394</td>
      <td>51.753489</td>
      <td>0.356686</td>
      <td>0.526027</td>
      <td>22.903527</td>
      <td>0.470196</td>
      <td>1.161452</td>
      <td>0.616274</td>
      <td>1.006748</td>
      <td>0.613026</td>
      <td>0.498970</td>
      <td>0.347313</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>94.000000</td>
      <td>126.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>96.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>48.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>211.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>133.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>97.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.50000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>130.000000</td>
      <td>240.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>152.500000</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>97.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.00000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>140.000000</td>
      <td>274.750000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>166.000000</td>
      <td>1.000000</td>
      <td>1.600000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>97.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.00000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>200.000000</td>
      <td>564.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>202.000000</td>
      <td>1.000000</td>
      <td>6.200000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>98.600000</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[8]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
Int64Index: 302 entries, 0 to 302
Data columns (total 15 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       302 non-null    int64  
 1   sex       302 non-null    int64  
 2   cp        302 non-null    int64  
 3   trtbps    302 non-null    int64  
 4   chol      302 non-null    int64  
 5   fbs       302 non-null    int64  
 6   restecg   302 non-null    int64  
 7   thalachh  302 non-null    int64  
 8   exng      302 non-null    int64  
 9   oldpeak   302 non-null    float64
 10  slp       302 non-null    int64  
 11  caa       302 non-null    int64  
 12  thall     302 non-null    int64  
 13  output    302 non-null    int64  
 14  o2        302 non-null    float64
dtypes: float64(2), int64(13)
memory usage: 37.8 KB
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[9]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[9]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>(302, 15)</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<ul>
<li>age: age of the patient</li>
<li>sex: 1 = male, 0 = female (binary)</li>
<li>cp: chest pain type (4 values) Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic</li>
<li>trestbps: resting blood pressure</li>
<li>chol: serum cholesterol in mg/dl</li>
<li>fbs: fasting blood sugar &gt; 120 mg/dl (binary) (1 = true; 0 = false)</li>
<li>restecg: resting electrocardiography results (values 0, 1, 2)</li>
<li>thalachh: maximum heart rate achieved</li>
<li>exng: exercise induced angina (binary) (1 = yes, 0 = no)</li>
<li>oldpeak: = ST depression induced by exercise relative to rest</li>
<li>slp: of the peak exercise ST segment (Value 0: up sloping , Value 1: flat , Value 2: down sloping )</li>
<li>caa: number of major vessels (values: 0–3)</li>
<li>thall: maximum heart rate achieved (0 = no-data, 1 = normal, 2 = fixed defect, 3 = reversible defect)</li>
</ul>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[10]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#displaying count display categorized according to gender</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">&#39;sex&#39;</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[10]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:xlabel=&#39;sex&#39;, ylabel=&#39;count&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Blood-Pressure">Blood Pressure<a class="anchor-link" href="#Blood-Pressure">&#182;</a></h2><p>Picture shows the range of blood pressure. Systolic mmHg if higher then 139 is considered to be High and if higher then 140 it is stage 2 of Hypertension</p>
<p>Let's set our Systolic limit to 139. We will consider patient in danger if having upper number higher then 139 mm Hg</p><p><img src="image.png" alt="image.png"></p>
<h2 id="Cholestrol-Levels">Cholestrol Levels<a class="anchor-link" href="#Cholestrol-Levels">&#182;</a></h2><p>Picture shows the upper limit of cholestrol of 240. We will consider cholestrol higher then 239 as in danger number of cholestrol </p><p><img src="image2.png" alt="image.png"></p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[11]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span><span class="o">.</span><span class="n">transpose</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[11]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>302.0</td>
      <td>54.420530</td>
      <td>9.047970</td>
      <td>29.0</td>
      <td>48.00</td>
      <td>55.5</td>
      <td>61.00</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>302.0</td>
      <td>0.682119</td>
      <td>0.466426</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cp</th>
      <td>302.0</td>
      <td>0.963576</td>
      <td>1.032044</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>trtbps</th>
      <td>302.0</td>
      <td>131.602649</td>
      <td>17.563394</td>
      <td>94.0</td>
      <td>120.00</td>
      <td>130.0</td>
      <td>140.00</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>chol</th>
      <td>302.0</td>
      <td>246.500000</td>
      <td>51.753489</td>
      <td>126.0</td>
      <td>211.00</td>
      <td>240.5</td>
      <td>274.75</td>
      <td>564.0</td>
    </tr>
    <tr>
      <th>fbs</th>
      <td>302.0</td>
      <td>0.149007</td>
      <td>0.356686</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>restecg</th>
      <td>302.0</td>
      <td>0.526490</td>
      <td>0.526027</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>thalachh</th>
      <td>302.0</td>
      <td>149.569536</td>
      <td>22.903527</td>
      <td>71.0</td>
      <td>133.25</td>
      <td>152.5</td>
      <td>166.00</td>
      <td>202.0</td>
    </tr>
    <tr>
      <th>exng</th>
      <td>302.0</td>
      <td>0.327815</td>
      <td>0.470196</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>oldpeak</th>
      <td>302.0</td>
      <td>1.043046</td>
      <td>1.161452</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.8</td>
      <td>1.60</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>slp</th>
      <td>302.0</td>
      <td>1.397351</td>
      <td>0.616274</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>caa</th>
      <td>302.0</td>
      <td>0.718543</td>
      <td>1.006748</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>thall</th>
      <td>302.0</td>
      <td>2.314570</td>
      <td>0.613026</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>2.0</td>
      <td>3.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>output</th>
      <td>302.0</td>
      <td>0.543046</td>
      <td>0.498970</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>o2</th>
      <td>302.0</td>
      <td>97.480795</td>
      <td>0.347313</td>
      <td>96.5</td>
      <td>97.50</td>
      <td>97.5</td>
      <td>97.50</td>
      <td>98.6</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[12]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">set_style</span><span class="p">(</span><span class="s2">&quot;white&quot;</span><span class="p">)</span>
<span class="n">matrix</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">triu</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">corr</span><span class="p">(</span><span class="n">method</span><span class="o">=</span><span class="s2">&quot;pearson&quot;</span><span class="p">))</span>
<span class="n">f</span><span class="p">,</span><span class="n">ax</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span><span class="mi">14</span><span class="p">,</span><span class="mi">14</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">corr</span><span class="p">(),</span><span class="n">annot</span><span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
            <span class="n">vmin</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="n">vmax</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span> <span class="n">mask</span> <span class="o">=</span> <span class="n">matrix</span><span class="p">,</span> <span class="n">cmap</span> <span class="o">=</span> <span class="s2">&quot;PRGn&quot;</span><span class="p">,</span>
            <span class="n">linewidth</span> <span class="o">=</span> <span class="mf">0.4</span><span class="p">,</span><span class="n">linecolor</span> <span class="o">=</span> <span class="s2">&quot;white&quot;</span><span class="p">,</span><span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">12</span><span class="p">})</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">rotation</span><span class="o">=</span><span class="mi">60</span><span class="p">,</span><span class="n">size</span><span class="o">=</span><span class="mi">14</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">yticks</span><span class="p">(</span><span class="n">rotation</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span><span class="n">size</span><span class="o">=</span><span class="mi">14</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Pearson Correlation Map&#39;</span><span class="p">,</span> <span class="n">size</span> <span class="o">=</span> <span class="mi">14</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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KCgoweNCQ0OxsbEx3s6aNSuhoaHx7s+aNStPnjxJcTY1KCIiIiIi/2P+2ZC8Lmtra8LCwoy3w8LCsLGxiXd/WFgYtra2Kc6mES8REREREXkthQoV4tq1azx8+JDIyEiOHj1KhQoVqFixIrt37wZgz549VKpUKcX70AqKiIiIiIgk6bvvviM8PBw3Nzc8PDzo2rUrBoOBFi1a4OjoSNu2bRkyZAht27bFysoKHx+fFO9LlxkWEREREUkj5nXzpneERMX+eCO9I7yURrxERERERMRkqEERERERERGToXNQRERERETSiJm5WXpHeOtoBUVEREREREyGVlBERERERNKIVlCSTysoIiIiIiJiMtSgiIiIiIiIydCIl4iIiIhIGtGIV/JpBUVEREREREyGGhQRERERETEZGvESEREREUkjGvFKPq2giIiIiIiIyVCDIiIiIiIiJkMjXiIiIiIiacTMXOsByaUjJiIiIiIiJkMNioiIiIiImAyNeImIiIiIpBFdxSv5tIIiIiIiIiImw2QblBMnTtCuXTvKlStH+fLl6dq1K8HBwQDs27ePxo0bU7ZsWbp168bYsWPx8PAwPnbHjh00bNiQcuXK0axZM/bs2ZNe34aIiIiIiCSDSTYooaGh9OjRg/fff5/vv/+exYsXExQUxNy5c7lx4wa9evWibt26bNy4kTJlyrBq1SrjYy9evMigQYP4/PPP+e6772jdujV9+/blwoUL6fgdiYiIiMj/IjNzM5P8MmUmeQ7K06dP6dGjB5999hlmZmbkzZuXTz75hBMnTrBu3TpKlSpF3759Afjyyy85cOCA8bGLFy+mRYsWNG3aFABXV1dOnz7NihUrGD9+fHp8OyIiIiIi8ppMskGxt7enWbNmLFu2jAsXLvD7779z6dIlypYty6VLlyhdunS8+vLly/Po0SMArly5wuXLl9mwYYNxe1RUFGXLlv1XvwcREREREUk+k2xQgoODadGiBSVKlKBatWq0bt2aXbt2cezYMSwsLBLUGwwG479jYmLo2rUrzZs3j1eTIUOGNM8tIiIiIvIicxMfpzJFJtmg/PTTT2TNmpWFCxca71uxYgUGg4EiRYpw+PDhePXnzp0jb968ABQoUIAbN26QL18+4/ZZs2aRPXt2Onbs+O98AyIiIiIikiImeZJ89uzZuXPnDvv37+fGjRssWLCA7du3ExkZSevWrTl79izz5s3jjz/+YP78+Rw9ehQzs7jutHPnzmzbto1ly5Zx7do11qxZw7x58+I1LCIiIiIiYprMDC/OR5mImJgYxowZw5YtWwAoU6YMH330EdOnT+fgwYMcOHCASZMmcfPmTT744AMsLS3JmTMnY8aMAWDLli34+vpy/fp1nJ2d6dWrl/GkeRERERGRf4t122LpHSFRoWsupXeElzLJBiUply9fJjo6mpIlSxrv6969O2XKlKFfv37pmExEREREJD41KMlnkiNeSbl+/TqdO3dm//793Lx5k3Xr1nHw4EHq1KmT3tFEREREROQNmeRJ8kmpXbs2v/32G8OHD+f+/fsUKFCA6dOnU7x48fSOJiIiIiISj6l/KKIpeutGvERERERE3hY27U3zTfQnqy6md4SXeutGvERERERE5L/rrRvxEhERERF5W2jEK/m0giIiIiIiIiZDDYqIiIiIiJgMjXiJiIiIiKQRMwuNeCWXVlBERERERMRkqEERERERERGToREvEREREZE0oqt4JZ9WUERERERExGSoQREREREREZOhES8RERERkTSiEa/k0wqKiIiIiIiYDDUoIiIiIiJiMjTiJSIiIiKSRszMtR6QXDpiIiIiIiJiMtSgiIiIiIiIydCIl4iIiIhIGtFVvJJPDco/PPzrUXpHSFL23NnSO4KIiIiISJrRiJeIiIiIiJgMraCIiIiIiKQRjXgln1ZQRERERETEZKhBERERERERk6ERLxERERGRNKIRr+TTCoqIiIiIiJgMNSgiIiIiImIyNOIlIiIiIpJGNOKVfFpBERERERERk6EGRURERERETIZGvERERERE0ohGvJJPKygiIiIiImIy1KCIiIiIiIjJ0IiXiIiIiEga0YhX8mkFRURERERETIYaFBERERERMRka8RIRERERSSMa8Uo+raCIiIiIiIjJ0ApKMuw7uI+5C/2IjIqkcMHCDB88Auus1q9d9+jxIyZPn8Tl3y+TOVNmGtVvROvmbgDsPbCXMRNG4+jgaHye+bMXkDVL1n/t+xMRERERSW9mBoPBkN4hTMnDvx4len/IwxDadm7DAt+FuLq44jt/NuHh4QzuP+S160ZP8MLCwoKhA4YRGxvLoBGDaNmkBdXer86cBXPImiULnTt0STJf9tzZUu17FREREZG0lWfwu+kdIVG3Jv+a3hFeSiNer+nwkcOUKF4SVxdXAJp/2oJtO7bxz/4uqbqLly5Sv04DLCwssLKy4oMqH7Bz904Azpw7zdHjR+nYvSPd+33OiVPH/91vUERERETEBJh8g7Jq1Spq1apFmTJlaNy4Mb/88gsAt2/fpnfv3pQvX56PPvqIqVOnEhkZCcDgwYOpU6eO8faWLVsoW7YsV69eTXGO4DvBONo7GG872DsQFhZGWHjYa9eVKlmKrT9tITo6mvDwcH7Zs5N7D+4BkM02Gy2btWT5guX0/rwPg0cOJvhOcIrzioiIiIi8jUy6QTl//jwTJkxg6NChbNu2jQYNGvDVV1/x6NEj+vTpQ7Zs2diwYQNTp05l165dTJs2DQAPDw+ePHnCkiVLePjwId7e3nzxxRcULFgwxVkMhthE77cwt3jtui97fYUZZrh368CQkYN5t/J7WFlaATBp7GQ+qv4xAOXLlqdsqbL8esx0l95ERERE5NXMzM1M8suUmfRJ8jdv3gTA2dkZZ2dnevToQZkyZThx4gRBQUEEBARgYRHXIHh6evLZZ58xcOBA7OzsGDp0KF5eXpw6dQoXFxe6dEn63I7EzF8yn7379wAQFh5GoYKFjdvu3ruLrY0tmTNnjvcYRwcnzl44l2jd7eDb9O3Zj2y2ceeRLF/9DS7OLjx58oQNm9bTqX1nzMzifmAMBgOWFib98oiIiIiIpDqTXkGpVq0aJUuWpGnTpjRu3JiZM2eSL18+goKCePz4MZUrV6ZChQpUqFCB7t27ExUVxa1btwBo0qQJ5cqVY+fOnYwbN87YyCRHj896sHLxKlYuXsVivyWcPX+W60HXAQjcHEj1D2okeMx777z30rrAzYEsWLIAgPsP7rPp+03UrV2PLFmysH7jen7ZEze+dum3S5y/eJ6q71ZN/kETEREREXmLmfxVvAwGA8eOHeOXX35hx44d3Lt3jy5duvD9998zf/78BPW5c+cmQ4YMhIeH06hRI27fvs3w4cNp3779a+3vZVfxAth/aD9+C+cQHRWNcx5nRg3zIpttNi5cPM+4KeNYuXhVknVh4WF4jRtF0M0gDBjo1K4z9T+pD8CFi+eZOmsq4eHhWFhY8FXf/lSuUDlBBl3FS0REROTtkXeYab7hfGP8wfSO8FIm3aCcOHGCAwcO0KdPHwBiY2OpX78+jRs3ZtGiRezevZts2eL+YD969CjLly9nypQpZMyYkQkTJrBv3z66dOnC+PHj2bJlC05OTq/cZ1INiilQgyIiIiLy9lCDknwmPeKVKVMm/Pz8WLt2LUFBQezcuZO//vqLcuXK4eLiwsCBA7l48SInTpxgxIgRmJubkzFjRs6ePcuKFSvw9PSkRYsWFC9eHC8vr/T+dkRERERE5BVMegUFYPPmzcydO5egoCAcHBz47LPPaN++PTdu3GDcuHEcOnSIjBkzUqdOHTw8PMicOTMtW7akYMGC+Pj4AHDp0iWaN2/OlClTaNCgQZL70wqKiIiIiKSW/CPeT+8IifrT+0B6R3gpk29Q/m1qUEREREQktahBST6THvESEREREZH/LfqgDRERERGRNGJurvWA5NIRExERERERk6EGRURERERETIZGvERERERE0oiFRrySTUdMRERERERMhhoUERERERExGRrxEhERERFJI+YWZukd4a2jFRQRERERETEZWkERERERERGj2NhYvLy8uHTpEhkyZMDb25t8+fIBcOHCBcaPH2+sPXnyJHPmzKFs2bLUrVuXokWLAlC7dm06deqUov2rQRERERERSSNv41W8duzYQWRkJP7+/pw8eZKJEycyd+5cAEqUKMGKFSsA2Lp1Kw4ODtSoUYMDBw7QqFEjRo4c+cb7f/uOmIiIiIiIpJljx45RvXp1AMqXL8/Zs2cT1ISHhzN79myGDx8OwNmzZzl37hwdOnTgiy++4M6dOynev1ZQRERERET+x/j7++Pv72+87ebmhpubGwChoaFYW1sbt1lYWBAdHY2l5d+tw/r166lXrx52dnYAFCxYkNKlS/P++++zefNmvL29mTVrVoqyqUEREREREUkj5iY64vViQ/JP1tbWhIWFGW/HxsbGa04Avvvuu3gNSJUqVcicOTMAderUSXFzAhrxEhERERGRF1SsWJE9e/YAcSfBPz/x/bknT54QGRlJ7ty5jfeNGDGCH3/8EYCDBw9SqlSpFO9fKygiIiIiImJUp04d9u/fT5s2bTAYDIwfP56lS5fi6upKrVq1+OOPP3B2do73mAEDBjBs2DDWrFlD5syZ8fb2TvH+zQwGg+FNv4n/kod/PUrvCEnKnjtbekcQERERkddUblLt9I6QqFNDdqR3hJfSiJeIiIiIiJgMjXj9g1YoRERERETSjxqUf3hy70l6R0iSTS4bQkNC0zvGK1nn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Xpbb54LTO8YrOZVy5PHdx+kdI0m29rYA3HyS8nNT/g3ONikbSxMREZH0F3BlZXpHSFTrQh3SO8JLmdSIl4WFBX5+fjx69IhmzZrh5eVFnz59aNCgQZKPc3JyYv78+ezbt4/GjRvj5+eHh4cHrVq1+peSi4iIiIhIajCpFRRToBWU1KEVlNSjFRQREZG3l1ZQks+kzkEREREREfkv0VW8kk9HTERERERETIYaFBERERERMRka8RIRERERSSMa8Uo+HTERERERETEZalBERERERMRkaMRLRERERCSNaMQr+XTERERERETEZKhBERERERERk6ERLxERERGRNKIRr+TTERMREREREZOhBkVEREREREyGRrxERERERNKIudYDkk1HTERERERETIYaFBERERERMRka8RIRERERSSO6ilfy6YiJiIiIiIjJUIMiIiIiIiImQyNe/1A8e9n0jvBKTqUc0zvCa7G1t03vCK/F2SZfekcQERGR/yiNeCWfGpR/uBV6Pb0jJCmPtSuPwx+md4xXss2SneCwW+kdI0mOWfMAcPXxxXROkrSCtsUJfxae3jFeKUumLOkdQURERP4D1NKJiIiIiIjJ0AqKiIiIiEga0YhX8umIiYiIiIiIyVCDIiIiIiIiJkMjXiIiIiIiaUQjXsmnIyYiIiIiIiZDDYqIiIiIiJgMjXiJiIiIiKQRc60HJJuOmIiIiIiImAw1KCIiIiIiYjI04iUiIiIikkZ0Fa/k0xETERERERGToQZFRERERERMhka8RERERETSiEa8kk9HTERERERETIYaFBERERERMRka8RIRERERSSMa8Uo+HTERERERETEZalBERERERMRkvDUjXsePH2fQoEHcu3ePZ8+esX37dvLly5dueQ7uPcwi38VERUVRsHABBnkOIKt11gR1BoOBSV5TKFCoAG4dWwEQ8SyCGZNmc+ncJWINBkqULs5XQ/qRMVPGN861b+8+5syeS2RkJEWKFGbEqOFYW1snu2bQgCHY2+disMcgAJ49e8asGbM5dfI0z54+pWnzJrh3ck9xzoN7DzJ/9iKioqIoVKQgQzwHJTh+L6t5/OgxPuOn8/vlK2TKnIkGn9ajRZvmADx+9JgZk2dx7eo1IiIicP+sA3UbfZLinC/6dd9Rls5ZTlRkFAWK5OerEf3Iap0lQZ3BYGDa6FnkK+RKS/dm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wIm9pzix95TxsQOmfcEH9aty9cKfjOrijSHGQPVG71Ppo4opzvMiOzs7Ro0cxeBhg4mKisLFxYUxnmOAuGM5dvxY1qxYk2Tdy5iZmTF9ynQm+0xm0ZJFmJmb0bljZyqUr/Ba2XLa5WTM6DEMHDTQuM9x3uMAOHfuHKNHjyYgICDJutatWhN0I4hWrVsRHRVNy5YtjeduzJwxk4mTJuI31w9LC0smT55Mjhw5yJEjB593+5yu3bpiaWGJvb39Sz/fKK0zAly7fg3nPM7x9rt06VKePXvGmtVrWLN6DQBWGaxYtXJVksfUzs4OT6+ReAwaSlR0NC4uzniNHcX5cxfwHjOO1f4rX1ojIiLyX1KnTh32799PmzZtMBgMjB8/nqVLl+Lq6kqtWrVwd3enXbt2GAwG+vfvT8aMGenVqxdDhgwhICCAHDly4OPjk+L9mxlesbzQuHFjqlWrRoUKFejXrx/nzp0zLvF4eHgQHR3N1KlT6dq1KzExMXh6ehIeHs706dPZt28fI0eOpEOHDri7u1OxYkX69+9P3bp1KV26NF999RXBwcGMHz+ec+fOMW/ePD7++GNGjBjBqVOnGDt2LBkyZGD48OGULVuW0aNHU7NmTXr16kWrVq0AOHz4MB07duTcuXPcvn2bWrVqsX37duMVgWbPns2BAwdYs2bNax2QfWtOpvhg/huqtS1P2MOwVxems6zZs7Lnrx3pHSNJNXLHNVahIaGvqExf1jmsefb02asL01mmzJlMPmemzHFjl4/DH6ZvkFewzZI9vSOIiEgqeRqe8svtpqXMWTK/uiidvHLEq3379qxdu9Z4Cc+XGTZsGM+ePaNZs2b07duXokWLUrduXeP5JS8aP348ly9fpmHDhnh4eFCvXj3Kly9vrB06dChlypShW7dudOnShdKlSzNkyJAUfosiIiIiIvK2eOUKyv8araCkDq2gpB6toKQeraCIiMi/TSsoyZdq56CIiIiIiEh8WgpIvmRfxUtERERERCStqEERERERERGToREvEREREZG0EqsZr+TSCoqIiIiIiJgMNSgiIiIiImIyNOIlIiIiIpJG9IkeyacVFBERERERMRlqUERERERExGRoxEtEREREJK3EpneAt49WUERERERExGSoQREREREREZOhES8RERERkTSiq3gln1ZQRERERETEZKhBERERERERk6ER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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 style="text-align:center;">Question # 1</h1><h4 style="text-align:center;"> Our main goal first of all is to search either patient had an attack or not?</h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[13]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">val_counts</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;output&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">no_heart_attack</span> <span class="o">=</span> <span class="p">(</span><span class="n">val_counts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">/</span> <span class="n">df</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">*</span> <span class="mi">100</span>
<span class="n">heart_attack</span> <span class="o">=</span> <span class="p">(</span><span class="n">val_counts</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">/</span> <span class="n">df</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">*</span> <span class="mi">100</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Heart Attack: </span><span class="si">{</span><span class="n">math</span><span class="o">.</span><span class="n">floor</span><span class="p">(</span><span class="n">heart_attack</span><span class="p">)</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;No Heart Attack: </span><span class="si">{</span><span class="n">math</span><span class="o">.</span><span class="n">ceil</span><span class="p">(</span><span class="n">no_heart_attack</span><span class="p">)</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>


<span class="n">sns</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;No Heart Attack&quot;</span><span class="p">,</span> <span class="s2">&quot;Heart Attack&quot;</span><span class="p">],</span> <span class="n">y</span> <span class="o">=</span> <span class="p">[</span><span class="n">no_heart_attack</span><span class="p">,</span> <span class="n">heart_attack</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Heart Attack: 54%
No Heart Attack: 46%
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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jzHGvP7662bXrl3GGGM2bNhgHn30UXPw4EEzZMgQc+zYMeP1ek12drY5cuSIueaaa0xWVpZZuHDhWX1+EThbt241AwYMaHDsXX/99Wbt2rVm8+bNvuPg+PHjZuTIkea3334z+fn5Zv/+/cYYY5566imzbNkyU15ebuLi4kxNTY0xxpilS5eagoKCBvuqr683Q4cONYcPHzbV1dUmPj7eHDt2zBhjTHJystmzZ48x5sRxW15ebgoLC337j42NNcYYc++995r333/fGGPMO++8Y7Zs2WKcTqd5/PHHTVJSkjl48GCAn7VzH2feZ+iCCy7QjBkz5HQ61a9fP0nSd999pz59+qhNmzaSpJiYGH3zzTe64YYbGjx26tSpio+P993+8/rx7t27tXz5cj333HMyxigkJETt2rXToUOHNHnyZIWFheno0aO+a4TR0dGNzlhYWKiffvpJd955pzwej8rKyjRlypTT3r9jx45atmyZQkND5Xa7FRERofLycvXo0UOhoaGS5Hv8wYMHVVZWpqioqDN52tDKBgwY0ODs9bHHHpP0+7FXWlqqlJQUSVJdXZ0qKip00UUXad68eQoLC1NlZaXvWO/cubPatm172v1s2bJFbrdbDz/8sKTfP9C3ceNGjR49+pT3b9++vXbu3KmtW7cqIiJCtbW1kqTvv/9effv2lSQNHjxYkvTaa6/pww8/lNvtbvTHcfxbcM27GQYNGqTo6GitW7dOktStWzd98cUXqqurkzFG27Zt8xvYv+rWrZumTJmivLw8zZkzR8OGDVNxcbH27dunRYsWafLkyTp+/LjvWntQ0Ik/NofD0eDa4aFDh7Rjxw6tWbNGK1as0MqVKzVkyBCtW7dOQUFBvvv+9et58+YpNTVVCxcuVM+ePWWMUVRUlL777jvfN1NqaqoqKyt14YUXasWKFdqzZw8vHv0DdOvWTXFxccrLy9OLL76o4cOHq0uXLpo5c6bmz5+vrKwsdezYsUnHniS98sormjt3rlasWKEVK1bo8ccf973u4nA4fNv587FFRUU6//zzlZOTowkTJviO8+7du2vnzp2SpA0bNigvL0+S9OCDD+qOO+7QnDlzAv7cnOuIdzM9+uijvrPSyy67TMOHD9eYMWOUmJioTp066cYbb2zytpxOp5588kklJyfL6XTqsssu05VXXqny8nKNGzdOqamp6tKliw4cOHDSY2NiYnT33Xf7vinWr1+voUOHKjg42Hef2267TQUFBerQoYM8Ho+ys7MVFRWl3bt364UXXtDIkSM1ceJEjR07Vnv37tWBAwfUoUMH3XXXXUpOTlZSUpJ69eqliy66SNLv33jz5s1TZmamDh8+/HeeRrSyQYMGKSwsTGPHjvX95M+IiAiNHDlS48aN0+233y63233KY693795atWqVtm7dKun3f5Xt2LFDAwcO9N2nf//+qqmp0aeffqq+ffvqkUceUVVVle+4vfrqq7VlyxaNGzdOs2fPVteuXXXgwAE98sgjWr58uVJSUrRx40aNGDHCt83Ro0frt99+a9a7pf5J+Hg8AFiIM28AsBDxBgALEW8AsBDxBgALEW8AsBDxBgALEW8AsND/A8Jh4NMsSrMmAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 style="text-align:center;">Question # 2</h1><h4 style="text-align:center;"> Based on vitals like BP, cholesterol etc. what’s the probability of a
patient to have an attack next year?</h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[14]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_new</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">output</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">index</span><span class="o">.</span><span class="n">values</span><span class="p">]</span>  
<span class="n">df_new</span> <span class="o">=</span> <span class="n">df_new</span><span class="p">[</span><span class="n">df_new</span><span class="p">[</span><span class="s1">&#39;chol&#39;</span><span class="p">]</span> <span class="o">&gt;</span> <span class="mi">240</span><span class="p">]</span>  
<span class="n">df_new</span> <span class="o">=</span> <span class="n">df_new</span><span class="p">[</span><span class="n">df_new</span><span class="p">[</span><span class="s1">&#39;thalachh&#39;</span><span class="p">]</span> <span class="o">&gt;</span> <span class="mi">100</span><span class="p">]</span>  

<span class="n">data1</span> <span class="o">=</span> <span class="n">df_new</span><span class="o">.</span><span class="n">loc</span><span class="p">[:,[</span><span class="s2">&quot;chol&quot;</span><span class="p">,</span><span class="s2">&quot;thalachh&quot;</span><span class="p">]]</span>
<span class="n">data1</span><span class="o">.</span><span class="n">plot</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[14]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXQAAAD3CAYAAAANMK+RAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABngElEQVR4nO2dd3hUZdr/v2d6pqQ3UoCEhN4JoAiIKIIFdVkUEHUta98o+v4UQQF9QV13FdcVfS3rriuwogirYgEFRUSQKgSSUFKAkN6TmWTqOb8/njnTMj3TMjyf6+IKmTmZeaac77mf73Pf98NwHMeBQqFQKH0eQbgHQKFQKJTAQAWdQqFQogQq6BQKhRIlUEGnUCiUKIEKOoVCoUQJonA++eTJk5GZmRnOIVAoFEqfo7q6GgcOHOhxe1gFPTMzE1u3bg3nECgUCqXPMW/ePKe3U8uFQqFQogQq6BQKhRIlUEGnUCiUKCGsHjqFQrl0MRgMuHjxIrRabbiHErHIZDJkZWVBLBZ7dTwVdAqFEhYuXrwIlUqFgQMHgmGYcA8n4uA4Ds3Nzbh48SJycnK8+htquVAolLCg1WqRlJRExdwFDMMgKSnJpxkMFXQKhRI2qJi7x9f3hwo6hRJhnKxux7GqtnAPg9IHoYJO6RN0ag3Q6IzhHkZIeGX7Kaz5qiTcw7jk2Lp1K1599VWvjj1w4ACeeOKJII/Id6igU/oEj338G5ZtPRHuYYQErcEErdEU7mFQ+iA0y4XSJ2jo1KG92xDuYYQEvZGF0XRpbSS25chFfHq4KqCPeVtBNn4/Icvl/VqtFsuWLUNNTQ0MBgNmz56N48eP495770VLSwsWLVqEBQsW4JdffsHf/vY3SKVSxMfH46WXXgroOAOJVxF6c3MzrrzySpSXl+P8+fNYtGgRbr/9dqxatQosywIA1q1bh/nz52PhwoUoKioK6qAplx4GE4tO7aVhueiMLPQmNtzDiHo2bdqEzMxMfPLJJ1i7di2kUilEIhE++OADrFu3Dv/+97/BcRxWrFiBdevWYcOGDZg4cSL+7//+L9xDd4nHCN1gMGDlypWQyWQAgJdffhlLlizB5MmTsXLlSuzatQsZGRk4ePAgNm/ejNraWhQWFmLLli1BHzzl0sFg4qA1XBo2hMF06UXov5+Q5TaaDgYVFRWYPn06AGDgwIGIjY3F8OHDwTAMUlJSoNVq0draCqVSibS0NADAxIkTsXbtWsyYMSOkY/UWjxH6K6+8goULFyI1NRUAUFxcjEmTJgEApk+fjn379uHIkSOYOnUqGIZBRkYGTCYTWlpagjtyyiWF3shCfYlE6HoTCwON0IPOoEGDcOIEWZepqqrC2rVre6QJJiQkQK1Wo6GhAQBw8OBBDBw4MNRD9Rq3gr5161YkJiZi2rRplts4jrO8aIVCgc7OTqjVaiiVSssx/O0USqAwmFio9UawbPRHrnojC8MlFqGHg4ULF+LixYu444478PTTT+Oee+7pcQzDMFizZg0KCwuxcOFC7N+/H4888kgYRusdbi2XLVu2gGEY7N+/H6WlpVi6dKld5K3RaBAbGwulUgmNRmN3u0qlCt6oKZccBhMLjgM0eiNUMu/6WvRVDCYOpkvgwhVupFIpXnvtNZf3/fDDDwCAKVOmYMqUKXb3T548GZMnTw76GH3FbYS+ceNGbNiwAevXr8ewYcPwyiuvYPr06ZadMvbs2YOCggKMHz8ee/fuBcuyqKmpAcuySExMDMkLoFwa8BHrpbAwSrJcqOVC8R2f0xaXLl2KFStWYO3atcjNzcXs2bMhFApRUFCABQsWgGVZrFy5MhhjpVzC8Fkf6kuguEhvpGJO8Q+vBX39+vWW/2/YsKHH/YWFhSgsLAzMqCgUGziOsywSRnuEznEc9CYWDGO/XkWheAOtFKVEPCaWA2e2lDu10V1cxFtLHAfqo1N8hgo6JeKxzfiIdsvFtqDISAWd4iNU0CkRj63IRbvlYrDxz2m1KMVXqKBTIh7bIptoLy6yFXEDXRwNKjqdDps3b8abb76Jjz/+2Ku/8aUjI4+rx7/iiit8ehxvoIJOiXhsBb0z2i0XI7VcQkVjYyM2b94c7mEEFNptkRLxGIxWYYv2RVHbCP2SSl889jHwW8/suV4x7g5g7CKXd7/zzjsoKytDUVERpk6diu3bt6OtrQ2PP/44Zs6ciQ0bNuC7775Dd3c3EhISsG7dOru/f+2113Dy5Em0tbVh6NChePnll9HS0oKlS5eis7MTHMfhlVdeAQDs2rWrx+Pr9Xr8z//8D2pqahAfH4+///3vXm8G7Qoq6JSIR38pWS40Qg8ZDz30EM6cOYNp06ahrq4OL774Ig4cOIB//OMfmDFjBtra2vDhhx9CIBDgvvvus/R9AQC1Wo3Y2Fj861//AsuyuOGGG1BfX4/3338fM2fOxKJFi3D06FFL59m0tDS7x585cya6urrwxBNPICsrC3feeSdKS0sxevToXr0mKuiUiMdwCS2K2gr6JdWga+wit9F0sBkxYgQAIDk5GVqtFgKBAGKxGE8++STkcjnq6upgNFq/e1KpFC0tLZb7u7q6YDAYUFlZifnz5wMAxo8fj/Hjx+PNN9/s8fgAEBcXh6ysLMvt3d3dvX4dVNApEY/domiUe+i2r/WSEvQwIBAILPs5OBZwnTp1Cjt37sTmzZvR3d2NefPmgeOsM6Y9e/agtrYWf/vb39DS0oLvv/8eHMdZOjgOHToUhw4dwu7duyGTyZwWiAWjaIwKOiXi4YVNwFxai6K042JwSUpKgsFgsETMtgwYMAAxMTFYuHAhACAlJcXSQhcARo8ejbfffhuLFy8GwzDIzs5GQ0MDHnroISxfvhxffvklAOCll17C559/HpLXA1BBp/QB9OZF0QS5JOoXRXW2hUU0Qg8qUqkUX3zxhd1tgwYNsrQ5+eijj9z+vatNfN555x27321botg+/i+//GK5/fXXX/d+4G6gaYuUiIeP0BMUkqhfFKWFRZTeQAWdEvHwgp6okET/oqhdhE4tF4pvUEGnRDwWQZdL0G0wRbUVcalludguNFJ64uv7QwWdEvHwi4OJSgkAQKOL3s2i7bNcolvsZDIZmpubqai7gOM4NDc3QyaTef03dFGUEvHYRugA0KE1IE4endvQXUoRelZWFi5evIjGxsZwDyVikclkllx1b6CCTol4bBdFgejORdfZVYpGt6CLxWLk5OSEexhRBbVcKBGP3mw9JJkFPZoXRm1tFtseNhSKN1BBp0Q8fCqfNUKP3lx0O8slyiN0SuChgk6JeHjL5VKI0PUm64Iv7YdO8RUq6JSIx9FDj2ZBt7VcaLdFiq9QQadEPHoTB4YB4mNIZks0L4rqjSykInJa0kpRiq9QQadEPAYTC7FAALlESBp0RXE/F52RhUJKks9opSjFV6igUyIeg5GFWMiAYRgopaKo7udiMLGQiQRgmOjPQ6cEHirovaBZrcO5Jk24hxH1GEwsxGYbQiUTR3ULXb2RhUQkgFgoiPpKUUrg8VhYZDKZ8Nxzz6GyshIMw+CFF16A0WjEgw8+iIEDBwIAFi1ahOuvvx7r1q3D7t27IRKJsHz58l5vpxTp/HXHaew61YBfl10NoSDwzeopBL2Jg1jIC7ooqhdFeUGXCAU0Qqf4jEdB//HHHwEAmzZtwoEDB/D6669j5syZuOeee3DvvfdajisuLsbBgwexefNm1NbWorCw0GW/4GihSa1HY6cOx6paMWFAYriHE7UYTCwkZkG/FCwXsVAAkZCJ6iZklODgUdCvueYazJgxAwBQU1OD2NhYnDx5EpWVldi1axcGDBiA5cuX48iRI5g6dSoYhkFGRgZMJhNaWlqQmBi9QtdtIMLyXUk9FfQgQkSOzIBUMhEa1bowjyh46E1Wy0VPLReKj3jloYtEIixduhSrV6/G3LlzMXr0aDz99NPYuHEjsrOz8dZbb0GtVkOpVFr+RqFQoLOzM2gDjwT4rn87S+rDPJLoho9aAUApE0d1hK4zktmIWEAjdIrveL0o+sorr2DHjh1YsWIFpk6dipEjRwIAZs2ahZKSEiiVSmg01gVCjUYDlUoV+BFHEF16IxgGKG/UoKJRHe7hRC16o72HHs156AY+QhdRD53iOx4F/fPPP8e7774LAIiJiQHDMPjTn/6EoqIiAMD+/fsxYsQIjB8/Hnv37gXLsqipqQHLslFttwBAl96EiQPJa9xZSqP0YGGX5SIVoSOKI3S9OUIXCRia5ULxGY8e+rXXXotly5Zh8eLFMBqNWL58Ofr164fVq1dDLBYjOTkZq1evhlKpREFBARYsWACWZbFy5cpQjD+sdOlNGJymREe3ATtLGvDA9EHhHlJUQhZFrR663shCZzRBKhKGeWSBxz5tkUboFN/wKOhyuRxvvPFGj9s3bdrU47bCwkK7Ha6jHY3OCIVEhGuHp2Hdj2Vo0eiRaO43QnEPy3J47+cKzBufiVSV+x1Z7Dx0cxWlWmuEVBl9gs6/ViroFH+ghUV+YmI56Iws5BIRrhmeBpYDfjzVEO5h9RnONWvw529P4enPijxuQWabh66URXc/F2uEztDmXBSfoYLuJ116IihyiRAjM+KQFivF9zTbxWu69CRDaPfpRnx5vMbtsaT037ooCkRvx0U+bVEkFNj1RqdQvIEKup/wgiSXCiEQMLh6WBr2nG2E1hC9GxgHEv59UkpFeGFbCVo0epfHkswPs4cujXJBNy+KSoQCGqFTfIYKugdO1XXgpW9Ke9gCGvOUXyEhAjNreBq69Cbsr2gO+Rj7IloDiT6XXjcUHd0GrPm6xOWxth66KtotF0uEzlAPPQxcbO3Csq1FaO/umx09qaB74K0fy/HengrUd9hXJ1oidAlZmLs8NwlyiZAWGXlJtzlCH5cdj4euHIStR6ux54zz3d8Ndh46H6H3zRPOE3yETptzhYe3fizDxwer8Ncdp8I9FL+ggu6GLr3RItDVbV0O9/GCTgRGJhZien4KdpbWg6VTZY/wgi4TC/GnmXnITVbg2c9PWNYmbNE7y3KJwgjdxHJgOZizXGiEHmpaNHpsPVoNlUyEjQcu4LcLreEeks9QQXfDztIGi/BUt2nt7tPwi6JSa+rcrOFpqO/Q4WRNe+gG2UfRmi+IMRIhZGIhXp43ClUt3Xj9+zM9jnXMQwei00PnF0H5PHRa+h9aPj54AToji4/unYQ0lQzL/3uyz30GVNDdsO14jWVj4urWbrv7us2CxHvoADA1PxkAcPR837uyhxpLhG6uAJ2cm4TrR6Xjv7/1zHgx2lguUnNKX7QLukhALZdQojey+Pe+c5iWn4xx/RPw/E3DUVrbgQ/3nQv30HyCCroL2rsN+Ol0I24Zl4m4GDFq2uwFnV8U5T10AEhWSsEwQEtXdPq7gYTPcomxef+yEuSW99UWvU3pP8MwUMnEUOui7z3m9xCVCBlIRH3LctEaTPho/zmY+qjd+M2JWjR06nDf1BwAwOwR6bh6aCrWfn8G1Q7nfiRDBd0FO4rroDexmDsmA5nxMT0+VMdFUQAQChjEx4jRoglte9cvj9fgiU+OhfQ5e4s1Qre+f3KJEN0Gk50ocBxnl+UCEB89KiN0k2OE3ncE/ZsTtVj5RTEO9MEsL47j8MHeSgxKUWB6fgoAEjg8f9MIsByH578sDvMIvYcKugu2Ha9BdmIMxmTFISM+pkeEzgs6v6EvT6JC4janOhh8XVSDL45V96noqNtgglQkgMBmpyfevuq2yeU3sRw4DhYPHYjeTS56euh95/Msre0AAJyu73stsw+fb8WJ6nbcOzXH7vuYnSjHkmsG4/uSenxXXBfGEXoPFXQnNKl12FfejLmjM8AwDLISYnp46HzrXKnI/i1MUkjRrA6toJc1qMFyZI/TvoJWb7KzWwDrAnOXje3C+8i2EbpKJgr5vqIVjWqnGTiBhI/IxUIBxCLGErH3BU7VESE/0wcF/YOfKxEvF2PeuKwe9903NQdD01V4/stip3ZgpEEF3QnfnqyDieVw09gMAEBGvAydOiM6bHKfNToTFBIRGMZ+L9EEhRitXaETdIOJxflmklLZ0Nl3BL3bYLKzWwCrfaXRWyN0vY3I8YR6X1GtwYQb/r4X7/xUEdTnsUToQgHEgr5VKWqJ0Ov6lqBXtXThu5I63D6pf48AAyDfuxd/NxI17VqnGViRBhV0J2w7VoP8VCWGpJENOjLj5QDsM126DUY7/5wnUSENqeVyvlljOfEbOrUejo4ctAa2Z4Rutlw0dhG6WdBFtoIe2kXR4pp2dBtMFtEKFjoHy8XEcn2ipqGxU4cmtR4ysQBn6tUem60FguNVbWgKwIz0w33nIGAY3HX5QJfHTBiQiEWT+uNf+86hOMJTkqmgO1Db3o2D51owd0yGJfrOiCftXW19dI3O5FTQkxQStHYZQnYiljVYd0pq6OhjEbrY/v3jPfQumwjdYJP5wRPqRdFjVeQkDvauVNbXSkr/AcDARr7twl/orhmWBrXOiJr24AYWXXoj5r+zDzP+uhvv/FQOndG//kmdWgM+OVSFG0b3Q3qc+xbOz8wZigS5GM99fjIkFyx/oYLuwNdFtQCAG0f3s9yWmRADAHaZLl16oyWitCVBIYGJ5ezsmWBiJ+h9yHLRGkyIEdt//Sweuo1XbTCSk0cksMlykZFF0VCdWMeq2gAAF1q6glpoYrsoKjFbTH0hF50X9JvHZgIAzgTZdmnrMsBg4hAXI8afvz2FWWv34NsTtT5/HzYfvgi1zoh7r8jxeGycXIzCmfn47UIbztRH7naTVNAd2Ha8BiMzY5GbYt3wOlkhhUQosLNcNDoTFFLnEToANIfIdilv1CAjToZ4ubhPWS7deu8idL1Ty0UEo7kffSg4VtUKibm3SlVr8HKS7QqLzBF6X6hUPFXXiX5xMkwyb8cY7EwXfnb27A3DsOG+yYgRC/HwxqO4+a1f8Ncdp/DjqQa0eVjHMrEcPtx3DgUDEjAmO96r571uVDoYBth+MnIzXjzuWHQpca5Jg+MX27H8+qF2twsEDDLiZfYRusGE+Bhxj8fgdyxq0egxKCW44wVIhD4oVYn6Dm2fsly0RhPiHN4/y6KoEw/d1nLhW+h2aA09LgqBplmtQ1VLN+aMSMf24jpUNKqRk6wIynPZZbmYI/S+kOlSWtuBoekqxMnFSI+VBT1C52e/KpkIU/OT8fVjU/HJ4Sp8cqgK7/xUARNbDgDIS1ViQv8ETBiYgAkDEpCbrLDYqDtL63GhpQvLrhvq8nkcSVXJUDAgATuK6/D4NfmBf2EBgAq6DV8VkbLzG0Zn9Lgvw6G4qEtnRIYT340X9FCkLrIsh/JGNRZMzAbH9S3LpVtvgsxhDYLP6XfmodtnuZhb6GqNSFUFd5zHL7YBAOaNz8T24jqUN6px9bC0oDyXbWGR2BKhR7blojeyKGtQY+bQVADA4HRVCCJ0Iuix5u+BSCjA4skDsHjyAHTpjSi62I4j51tx5HwrthfX4ZPDVQCABLkY4/snYPyABHxXUo/M+BjMGu7bZzl7RDrWfF2KC81d6J8kD+wLCwBU0G3YdrwWBQMSkBkf0+O+zPgY7Dlrbe/apTc59dB5QQ9F6mJthxZdehPyUpVo7zKgskkT9OcMFFoDixixq7RFJ1kuDpWiQGgadB2raoeAIX16EhUSVDQG7z3W2aYtWjz0yI7QyxrUMLIchvaLBQAMSVPi3xXNMLEchALGw1/7R0c3+dz5Rm22yCUiXJabhMtykwCQoKeiSYOjZoE/fL4Fu8xbRT53wzCIhL65zryg7yiuw/3Tc3v5SgIPFXQzp+s6cbq+Ey/cNMLp/RnxMWjo1Fn2fOzSG5166LaWS7DhF0TzUpS40NKFxk4dOI7rkRsfiXQbTD0EXSoSQMAAXTobD93Ys7CI74nemxa6p+s6sXRLEdbfN8kS8TvjWFUbBqepIJeIkJusCKqgG2xL/wO4KFrV0oU/ffwbPvhDAZKV0l4/ni2n6siC6PB+ZKo0OE0FvZHF+WaN3TpUILFE6E4sT0cEAgZ5qUrkpSpx28RsAECrRo+zDWqM7x/v83NnJ8oxIiMW2yNU0OmiqJmvimogYIDrR/Vzen9mQgw4Dqgzp2RpnFQ6AqS/t0IiDInlYhH0VCVSVTLoTSzaIrQx2P7yZrvFJLIoav/1YxgGConIedqiyMZDD0AL3R9PN+BYVZulKMsZHMfheFUbxplP/NwUBSqagpfhYFtYxK8ZBCJCP3K+Fcer2oJS9FNa2wGpSICBSWRdYUg6EfZgZoJ0aF1H6N6QoJBgUk6iz9E5z5wR6ThyvhUNHZGXhEAFHeTE/fJ4DaYMSkaKynkEw9swF9tI6preyNq1zrUlQSHpteVS1tCJdg/iXNagRoJcjCSlFKnmcUeqj/7GrjOWXWA4joPW2DNCB0jqYpcHy0UlJZFZb3Yt4kvU3V0UzjV3ob3bgDFZ8QCAQSlKNKn1QduezLF9LhAYD53PfgrGxb60thOD01QWccxLJVF5MFsAdGgNkIoEkIqCuyDuijkj0wEAOyJwdzIq6ABOVLfjfHMX5o5xHp0DVkGvadOiy9Cz06ItSQpJr9IW9UYWv3t7H5767Ljb48ob1JYTyCrokRc1AEBlk8YSWemMLDgOPRZFAZK6qPG4KNp7y+WsOYJ015/jWBXpaz/WEqGT99pVgdE/91Zif7n/3Qbte7kELsuFz35q6w78rPFUXQeG9bOuTMslIvRPlAd1YbSj2+jWJgs2ealK5KYoIrJhl0dBN5lMWLZsGRYuXIhFixbhzJkzOH/+PBYtWoTbb78dq1atAmuuZlu3bh3mz5+PhQsXoqioKOiDDxTbjtdALGQwe0S6y2P6matFq1u7LR6vs0VRgO+46H+kfPRCKzq1RnxfWo9yN9WJZY1qDDKLTGosGV8kpi6qdUbUd+gsEbWlF7qrCN1GZPVOmnMperkoyrKcxa5yd1E4XtUOuUSIfHMqTW4KsRXKnfjoTWod1nxdgg/3Vfo1JsAaoYuFDMSCwFku/Kwt0BF6Q6cWTWo9hqbH2t0+OE0V1NTFTq0BsTHhW/5jGAZzRqRjf3mzx3z3UONR0H/88UcAwKZNm7BkyRK8/vrrePnll7FkyRL85z//Acdx2LVrF4qLi3Hw4EFs3rwZa9euxQsvvBD0wQcCluXwVVEtpuenIF4ucXmcVCREikqKmrZuSxaGs0VRwNzPpRce+t6zTRAwxEv9x8/OG0K1aPRo0eidROiRJ+jnzNk3WgOxqmz3E3VELhbZZ7nY+Mo8EpEAUpHA7wj9Ymu3ZQzuujb+VtWGUZlxlmyN/olyiAQMKhpJvxKtTZvfnSX1YDn7yl1f0ZnIgjvDMJYIPZCWS6CtolO1RLSH9bMX9CHpSlQ2afwuyfdEh9ZoSVkMF7NHpMPIcthV2hDWcTjiUdCvueYarF69GgBQU1OD2NhYFBcXY9KkSQCA6dOnY9++fThy5AimTp0KhmGQkZEBk8mElpaW4I4+ABw+34radq2ls6I7+I0uuvWeInQxWnpx5f65rAljs+Mxf0IWthypdmqj8MIxyCzoCqkICokwIi2XCpt0yk6tAVoDEWnXHroTy0Vkn7mjkon9jtBt/V1XlovOaEJpTYfFbgHILKF/ohxbj1bjspd3YeKanZZF8u3m6ff55i6/o2qDkbNcuES9iNAPn2tx2hIi0NEkX/Jva7kAJEI3slzQ0mg7ug1+L4gGitFZcegXJ7N87pGCVx66SCTC0qVLsXr1asydO9cuNU6hUKCzsxNqtRpKpTVNib890tl2vAYysQDXeFEskmne6MLZ9nO2JCqk0BpYv/pnt3XpceJiG6bmp+CP03JhYMleh47YpizypMbKIjJCr7SxKDq0RssF0VmErpCInHdbdMhIIC10/Ys4z5rfO4aBy40ySms7oTexGGteEOW5bFASNHojxmUnoNtgwjs/laNDa8AvZU3IiJPByHI43+yfkOlNJkjMkbmnPPSvimrwxs6zPW4/XdeJ2/9xAC9/U2q5rbEjOJYLX/LvOLPlM12C1UqXWC7hjdAZhli0e840RlSfdK8XRV955RXs2LEDK1asgE5nFQ2NRoPY2FgolUpoNBq721WqIJfx9RKjicU3J2px9dC0HjsPOSMzgUTovCXgblEU8K9adF95M1gOmJafjJxkBeaMSMf6/ed7fGnKG9WIEQvtiqBSVFLLyRtJ2Kb6dWoNFrvDWdqnXCK0CD7g3EMHiKD7a7mcrSdCpJS6foxjF+wXRHle+t0oFK26Fu/cOQHzJ2ThPwcv4OMDF2AwcXjAnJfsr+2iN7KWClFe2F3loX9xrAYbD5y3u01nNOHxTb9Bb2RRab6odOtNFlupLcCWC1/y70hushIiARO0TBdiuYS/hGbOyHTojCx+OtPo+eAQ4VHQP//8c7z77rsAgJiYGDAMg5EjR+LAgQMAgD179qCgoADjx4/H3r17wbIsampqwLIsEhMTgzv6XrK/ohnNGr3b7BZbMuJk0BlZXDQ3aHJ1EUjoRbXoz2eboJSKMNbcMOiB6bno0Bqx6VCV3XFlDWrkpijstsxKVUkj0nKpbNJYLn4d3Ua3i6IKqfMsF4mDoPemhe6Zhk7kp6mgciPoxy+2I1UlRXpsz/YO/Oz00avywLIc/rLjNFJVUvx+Atnxxl9BN5g4i5DzlovRRfvcFo0ebd0Guw6Da787g1N1nRiVGYeLLd1gWc7u++ApDdYX+JJ/R/8cIBejnGQFTtcFJxe9U2sIu4cOABMHJiJJIcGOCLJdPAr6tddei5KSEixevBj33Xcfli9fjpUrV+LNN9/EggULYDAYMHv2bIwcORIFBQVYsGABCgsLsXLlylCMv1d8eawGSqkIM4akenV8ZgLp3cBHHq4tF/87Lu4ta8RluUmWiHRc/wRMyknEP/dW2k2/y2xSFnlSVZFnuXAch8pGDUZlxgEwR+gWy6Xn108ucchDt8n8sEUl829fUT7DJT9VaWnD64xjVW0Ymx3vtuo2O1GOWwuyYGI5XDsiDSqZGBlxsl5F6PyFy9Kcy0VHyVaNHnoja1mPaNXo8d7PFVg4MRsLJ2VDb2JR16G1fB/6xckCmrbIl/w7E3SA9HQJRoTOv+Zwe+gA2RR+1vA0/FDaELQFYF/x+K7I5XK88cYbPW7fsGFDj9sKCwtRWFgYmJEFGZ3RhO3Fdbh2RJrXHfv4jS74HGZXi6K85eJrpsv5Zg2qWrrxx6n2JcUPTs/Fff8+jK+LanHLuEx06Y2obuvGwpRsu+NSY6Xo0pug1hkt/U7CTaNah06dEWOy43GgsgUdWoNlZuN0UVQihMHEWVosGEwsGAY9+oIopWK/LJeq1i5oDSwGpylx9EKr08do69KjskmD+RN67jHpyJ/MPbIXFPQHQBapy/zcCENnZC1Czv90tQ0dHyy0dxsQIxGivlMLjgOm5adYUvoutHRZbL/8NBUOVPifI++IqwVRnsGpKnxzotblvgH+4kvZfyiYPSIdmw5VYV95M67yMjAMJpdsYdGeM03o1Boxd4zn7BaeLPNWdHwE5ipCd2a5aHRGvPbdabdX8j1nmwCQRlC2XDUkFXmpSry7pwIcx1n6ifSM0M2pixFUkswviPLVlh3d7hdF5Zae6ERo9SYOYqGgR6Sskon82kSEL0nPT1O59NCPXyQ7FI3zok92ZnwMti+ZjlFZZAaSl6pEeYPGrx2rDCbWsum42E3pv9HEWlIQ+ai7VUN+T5CLMSCR5MtfaO6yWC6DU5XQGVm7VMvecKrOvuTfkSHpSnC9TON0Rm/L/gPNlLwkKKUi7IiQHumXrKBvO16DBLkYU/OSPR9sJjaGpAY2a/QQChjLydfjOJkIYiFjZ7nsLWvCmz+U4fC5VpePv/dsIzLiZMh16LctEDB4YHouSms78PPZJrseLrakqszFRRFku/Cpa6My48AwfNqi60VRPref99ENJraHfw5YF0U5jkNJTQeavdxf8mwDsQHyU5U9Flaf/OQYcpd9jT/88yAYBhaR9oW8VCW6DSbUtPu+EQY/KwHgtjlXq40XzvvifEpivFyCjHgZhAIG51s0aOjUQSRgMND8nQpUpotjyb8jg9OCk+ni2Do33EhFQswcmorvSuphioD9X/usoPdm+7EuvRHfl9TjulH9emRPuINhGMt2dHKx0KW/yjAMEuQSO8ul1SzujS7E1mhisa+8GdPyU5w+7s1jM5CqkuK9PRUoa1BDKGAwwCE6So2NvOKiiiYNJCIBMhNioJKKSNqiu0pRc4TebY7QjSa2h38OkEVRjiPR9E3r9uLqtT/hi2PVbr8X55s12LD/PHKTFVDJxFBI7D30oxdaMThNhcKZeXj9trF+lZfzaaT+RKZ6k9VykbhJW7Sd+fGZK7zIJyjEEAkFyIyPwYWWbjR06JCiklrWdQLlozuW/DsyIEkBiUgQcB/d2jo3MgQdINkuLRo9Dp0Lf91NnxT0v2w/hfs/Ouz33+8qbUC3wYS5Tjay8ESGOU1Q7qJKlCfRoZ8Lf8K5EvSi6nZ0ao097BYeqUiIe6fmYG9ZE745WYsBiXJLNMcTiZZLRaMGA5PkEAoYqGRidNgUFjnNQ+cjdHN7Bd5ycYQ/oZ/ZUgS5RIicZAUe33QMD6w/4vT1n2vSYOF7v6LbYMK628cDIG14bdNBWzR6XJabhP+5dghuGZfp1+vlC738EXSDyTZCd70FnW06LB+h8yKfYM4JH5Akx4VmDRo6tUhVSS27awUiQndV8m+LUMBgcJoS35fUo9aP2YorrB56ZFguAHDl4BRIRYKIyHbpk4IeLxdjZ2kDzvp59d92vAapKikm5fieVsnnfbvqtMiT6NBxkZ8Su0or3Hu2CQwDXOHGArp9cn8opSJUNGoswmFLXIwYEpHA5UUjHFQ2Wbdsi40REw/dYIJEKHC6AQIfofO5/gabqNUWvif6qbpOPDlrMD57aAqevX4Y9pxpxKzX9+C/v120ROu8mGsNJmz842UYnkGESCUVQa0nto3RxKJDa0S8vHeRX5JCgni52Gm/F0/YZrnwaYt8Hr5tpG77veK99FaNHjKxwHKR7J8ot/TIT1HJECcPnKC7Kvl3ZPl1w9Ck1uOWt37BCfO6RG+xbj8XORG6QirCtPwU7DhZF7KNy13RJwV93vgsiAQMPnHIzfYGo4nFL2VNuGZ4ml87qvARujP/1xbSoMs2Qndvufx8thEjMmItU2NnxMrEWDSJZLY4+ucAsXpSlNKIsVyMJhYXWrosXQr56k5nvdB5LBtF62w8dCdrFfy+ooPTlLjjsgEQChjcPz0X3zw+DXmpSjzxyXHc/9FhHKhoxsL3foXexOI/91vFHCAnIseR3ad468Ld++8NDMMgL0WJcn8sFyNr6eHCMAzEQgZGE4uatm6MWLUDR86TKb3tzM+yKNplQKJNxWb/RDlauww439yF1FippZqzPQCWi6cMF54pecn47OHLIRIIcOc/DwRkw2u+9iASCotsmTMyHTXtWpyoDsyFy1/6pKAnK6WYNTwNW3+r9jn/82RNBzR6Ey43b1HlK1kJ3kXoSQqJ3UKdxXJxsnin1hnx24U2TMv3vKv0vVNzkKKSWrbYciQ1NnKKiy62dsNg4qwRukyMDi0pLHJ1QYxx2IbO4MJDz01RIF4uxv/ePNJuYW5QihKfPng5Vtw4HHvLmrDAIuaTe0SUtjsf8Wsc7hq0ecvQfiqU1Hb43IdFb2IhtXktYiFJ2yxvVENvZFFkjnL5sapkImu2S5febuwDzPtddhtMAbdcSms7nJb8O2NoeiweuWoQ2roMTr/7vtLRbQDDeD7/Qs01w1IhFDB2m7iEgz4p6ACwYGI2WjR67CzxrdsZ36/alSB6wlsPPUEhQYfWaDmpLZaLk9L8X8ubYWQ5TPMi46ZfXAwOLr8aVw52Lv6pKmnEtNDlM1xyLYIuspT+u8r95z10vkGX3ujcQx+QpMBvK2Y5/RyFAgb3Tc3Bt49Px+LJ/bHpgcuc+r22e5PyF9zEAAj6FYOSodYZcbyqzae/s81yAYjtYjBxls+z2lyh3KLRQyUTIUUptQh0a5ceCQqrDZGdaN3AOFUlg1wihFjIBKT8/1Rdp0e7xRa+2rY+AN/LDq0RKqnIrkI6EoiXS3B5bhK2h9l26bOCPi0/BRlxMmw6dMGnv/u1ohl5qUqXOxN5wlsPPckhF91dhL63rAkysQATBiZ4NQZ31YuRVC3K93LnLRfioZO0RWcZLoBtHrrVcnGVGudp79ScZAVe/N0oSwqdI7yga3RGiz3WWw8dAKYMSoaAIW0cfEHvsF7AF1bx3xk+FbJFo0eiQoLYGLFNhG5wiNCtGVCpKikYhkFcjKTXEbrOaEJZg9ppDxdXpJkFne9M2Rs6tIaI8s9tmT0yHRVNmoDn3vtCnxV0oYDBrQXZ2FvWhKoW1/tC2mIwsTh8rgWX5frfYyZVJYVQwHjhoZMLBi8UfITe1mXoYRP9fLYRk3KSArKlVqpKivZuQ8AKSHpDZZMGcTFiJJhFUiUToVNnRJfedYTOF2vxm1yQPPTgRGO8oKt1Rsvn01sPHQDi5GKMyorHz2d9a9pk6BGhC2B0EaEnmhdfLYuiXXrL+wyQ18YHFXw6Kzm+dx56eYPGbcm/M3hBD4QV2Kk1RkyVqCOzh6eBYRDWbJc+K+gAcGsBKc3efOSiV8efqG43++feFxM5IhIKcPOYDI8efKJN+T/HcWjtMjjtwljT1o3yRg2mu0hX9BX+5PUn08VgYl1ur+YPlU0a5CQrLJF0rEwMjgOa1HqXEbpYKIBEJLArLPKlVsAXbHc+anFI++st0/KScfxiu0/VrHqHBWCxiLGL0KvbbARdLkF8jBhtXQaYWA5t3YYeY+dtF77gjD++N3i7IGpLkkICkYDxKkKva9di7pt7XYpiJPRCd0VqrAzjsuPD2iO9Twt6VoIc0/JTsPlwlVdVWr+ae1lM7kWEDgBrF4y1dNZzhUXQu/To0BphYjnL1N/WEtnrotzfX/ytFjWxHP70n6O4Zu1PAavuq2zSWLZtA6zl2g0dWrczHIVNgy5XeeiBgB+PRmdEW5cBMrHA48zLW6blJ8PEcl7vMVrWoIbBxNlVQIoFAhhYDo3myLZJrYfWYLJE6HExYrR16dHRbQDH9bwYDUiSg2GAZCW5PV7ee0H3VPLvDIGAQapK6pWH/vr3Z3Ciuh3/8+lxpxtkdEbAbkXumDMyHSerO7x2DQJNnxZ0AFg0MRu17Vrs8WJ6+2tFC/JTlUhW+uef+4JF0DV6y3R+cBrxkm2j55/LmpCikmKIC5/XV/i1gUYfprccx2H1VyXYUUy2Udty1LsZjzu69EbUtmvt2hjwU+WWLr3LtEWA+Oh8YZHBGLwI3dZy4aPeQDGufwLkEqHlgu2Jl78phUoqssw6AXOWi5FFQ6fOkmJb3daNli6zoMsl6NQZLWmMtouiAHDDqH5YUJBtWYOIi5H0ehs6TyX/rkiNlaHeQ8Hb2fpObD5ShbljMiAUMHh049Ee1mGH1hBxKYu28PsSh8t26fOCfvWwNCQpJPjkoPucdN4/v3yQf9ktvsL7mc1qvWVBdLB5IYkXdJbl8EtZE6bmJXtc4PMWf8r/Nxy4gA/3ncN9U3Mwa3ga/vtbda9zhvnoKifZmi/PR8Qc57xKlMe2hS7JQw+Oh66Q2qctBiJlkUciEuCy3CSvfPS9Z5uw61QDHp2ZZxdsiEUMjCyLxk6d5YJ/tr4TeiNLPPQYYmHx0aDj+K8dkY4//3605XcSoffOQ/dU8u+KdC8E/ZXtp6GQiPDCTSOw9rYxKKntwP9+VWJ3TCR76ABZjB7WL5YKur9IRAL8fkIWdpbWu/WNiy62o0tv8jtd0VdEQgHiYsRo7dJbMl343eP5xaGS2g60aPSYFiC7BQCSFFIIGOfpka748JdKjO8fj2evH4bfj89CY6cOe8t8y9BwxJKyaGO52E6VXXnoACC32eQimB66VCSAWMgQQTdHvYFkal4yzjV3oabNdem7ieWw5usSZCXE4O4pA+3uEwkE6NQa0ak1WnZO4nPRE8yWC2B9rz35//ExYmj0Jpc91j1xsrodTWq9TwuiPGmxUtS5EfRD51qws7QeD80YhESFBFcPS8OD03PxnwMX8MWxagBkJtkZ4RE6AMwZkY7D51vDUrHd5wUdAOZPyIKR5fBdieurosU/96Pc31+SzP1c+KiIb5LEf9B8WpsvHR89IRQwSFZ6X1x0obkL5Y0a3DA6AwIBg6uGpiBeLsaWo9W9Ggff4tfWa7VdzHIn6AqJ0NKcyxBED51hGNJC15yHHoiURVvGmNvvltR0uDzmsyNVOFXXiWeuG9pj1iIRCiypiqMy4yAUMJZKRL7FAGAr6O7Hzx/vj+2iNZjw5KfHkKKS4uaxvve5SYuToVNrdLrPLsdxePmbUqSqpLj3ihzL7f9v9hBMGJCA5VtPoLxRDY3eBJaLrLJ/Z8wZmQ6OA74vqQ/5c0eFoOenKpEWK8WvFa67nf1a0YwhaSokhcA/50lUkI6Ltr2qU5RSi6DvLWvEkDQVUp1sc9YbSLWod9HBj6dJYdbMoaQ5v1QkxE1jMvBdcZ1f/cZ5Kps0yIiT2S0y2k6V3S0+2nrojrnZgUYhJQ26ghGhWzZLdtFzSK0z4tXvzmDCgATcMKrnNogioTUzJD1OhvRYmUXQE2wE/Zx5/1BPllFcL8r//7L9NM7Uq/HqrWP8ep/SVK6Li74rqcfRC214YtZgu++FWCjAutvHQSIS4NGNRy1N1yKpMZczBqcpMTBJHpZsl6gQdIZhcFluEn6taHZapaU3sjh8rrVX+ef+kGDu59LWpYeAIZZDioqIrdZgwqFzrQG1W3hSVTKvLZcfTjUgJ1lhKc8HgN+Pz4LOyOLrolq/x1DRpEFOin0mhG2E7s5DV0gdPPQg5aEDZGG0vduAdidpf4F47KyEGJxykTX07k/laOzU4bkbhjldQyGl/+T7nKKUIjMhxpKlkii3Wi7nmjUQChiPVoS/5f8/n23EP3+pxN1TBrqsUPZEehwv6PYzR6OJxV+2n8KgFAVudZI51i8uBmsXjCWzmC0nAER+hM4wDGaPTMe+sians6HKJg3u+/BQULatiwpBB0gpf2OnDhVOUp1OVLeh2xA6/5wnSSFBS5ceLV16xMWILelbjZ06HKxsgd7IBixd0ZZUlXcRepfeiP0VPbfOGp0Vh7xUJbb6me1CdlVSIzfZvoGYVCS05Fl7WhS1eOhBzHIByEWmuq3bnPYXeKEYmq7C6bqelktNWzfe21OBm8dmYFx/5xXCtj1sUlVSZJmrlAEgUSlBXAy5AFW3diNBLva4sB7vR8fFVo0e/2/zceSlKvHMdUO9/jtH0syL9Y6CvvnIRZQ3avD0nKEuM2euGpKKR2YMwkFzv/FITlvkmTMiHUaWww+netouu0rrsetUg6UBXSCJGkHnC32c5f3yVszkEAt6okKCVg2xXPjoL0UlRaNah5/PNkIiFGByTuDHlKqSolmj85ipsq+sGXoja7FbeBiGwbzxmTh0rhXnm31rA8uyHE7Xd6JTa7SL+nn4k9HtoqhEZFMpylk6EAYDhVRkyRJJCLDlAhDbpaJR02Mh8q87TgMAnp7jWiT5C5mAAZLMETq5nYFKKrJE6CznXVOx+Bh+kwvvBJ3jOCz/7wm0aPT424KxXu+964y02J4RerfehNe/J5bTtcPT3P79k7MGY9JAMsOO1MIiW8ZkxSM9Vua0WVd5oxoJcnFQvm+R/854yYAkOdJjZfi1ohl3XDbA7r795c0Ymq4KuEfqiUSFBEaWw4WWLsuHl6KSQm9k8c2JOkwYkBCwQhZbUmJllopMfqrrjB9ON0AhETrtC/+7cZn4647T2HK0Gk/OGmx3n4nlUNvejXNNXTjXrMG5Jg3ONXfhfLMG51u6LOLlrN9HrEyEJrUOMRLXIq2QCNFlMIFluaB76EqbjJpAWy4AMCQ9FkaWQ3mj2pIdcryqDf/9rRqPXjXI0hvIGXzEmqgg7Sb4YxPkEjAMA4mIMad4mryaXVh7onvnoW85Wo1vT9bhmeuGYmSm79vx2aKUiiCXCFHXbp05/vOXSjR06vD24vEeZxcioQDrFo/Dxweq7FogRyoCAYPZI9LwyeEqdOvtu4uWN2os/Y0CTdQIOvHRE7G3jPjo/BdEb2Rx+HwLFk7sH/Ix8ReQika1Jf+dL/ypbuvG4suCMybLzkWdWpeCrtGRbfim5ic77TfeLy4GU/OSsfXoRSy5Oh8CAYPjVW34975z+OZkrWXXIQCWysGcZAVmDk3FgCQF8tOUKBjQ00pQmaNKmZu+NXJzn3K12UcPpoduG+0F44LPX9ROmzsUchxJU0xWSvDwjDy3f8tbLvx3ho/QbccZHyNGl97kVYSukoqQFivFj6cb8MdpuW6PvdDchVVfnMTknETc7+FYb2AYhuSim7OvWjR6vLO7HLOGp6FgoHdrW6kqGR6/Jr/XYwkVs0ek49/7z+OnM42YMzLdcntFowZXDfFvLcITUSPoAPHRPz9Wg/JGDQalKFDWoMbnx6qhNbAhXxAFrCeexuaEs+3yOC0vOB+qdSs61z766q9K0KTWuT2xfz8+C0s+OYbXd57B3rIm/HahDQqJEL8bl4XRWXEYmKTAwGQ50lQyr9uZ8gt3Mg+l/wBQak73C2qWi03XzECnLQKk46NYyFgWRrefrMOhc614ed4oS6WqK8QC8rr5z5OP0G0FPU4uQU271qsIXSBgcO8VOXj521M4XtVmSat0xGhi8eSnxyAQMFi7YKxfG8E4IzVWinpz1s66H8qg0Rvx9OwhAXnsSGRSTiLi5WLsKK6zCHqH1oAmtY5G6N7AR8HPbClCbbvW0sxofP94TPVi84hAk6Swijd/wvG9VhLkYowI0tQxNdZ9P5cdxXXYdKgKj8wYhIluoqNrR6RBIRHizR/KkJOswKq5wzF/Qlavsgy89dABYNH7v0IuEQZ1MVsZ5AhdLBRgUIoSp+s6oDOa8PK3pzA0XYXbCrI9/63IPkLne/Hbeq9x5hQ+b/3Y2yf3x7ofy/DOT+X4vzsmOD3mnZ/Kcfh8K95YONatJeQr6bEyHLnQiqqWLqz/9RxuK8hGfoBaXkQiIqEAs4alYXtxnaXXPV+fkZvifS8cn57T3Z0GgwHLly9HdXU19Ho9Hn74YfTr1w8PPvggBg4cCABYtGgRrr/+eqxbtw67d++GSCTC8uXLMXr0aHcPHRT6J8qRl6pEaW0HrshLxqNX5eHKISkB/VL6gm1vDccI/Yq85KA16U9VSREXI8bbu8swcWCC3UnT0KHFM1uKMDIzFkuuGezmUYiw/uMPE6E3sZgWoPHyFoc7QR+YrADDkK0Gn5o9xLKgFgz4KFkqErgdU28Ymq7CwcoWfLTvPC60dGH9fZO8inpF5gid/87IxEIMTVdhcKr18+QXOr31/1UyMe66fADe3l1OMpEcIsXjVW34286zuGlMhl8FRO5Ii5WhvkOH1747DaGA8fj9iwbmjEzH5iMXsb+iGVcOTrF0Mx0Ujgj9yy+/RHx8PP7617+ira0Nt9xyCx599FHcc889uPfeey3HFRcX4+DBg9i8eTNqa2tRWFiILVu2BGXA7mAYBl8/NhUMGKe+cKixj9DJCRcrE+GuywfgpjEZQXtesVCAj+6dhD9+dBjz3t6Ht+8Yj2n5KeA4Dk99VoRugwl/WzDOq/co0L1v+OIid4vBEwYk4Mya64JqtfDwgs4vNAaDIemx+PxYDd7YdRZXDUnxaqtBAJbPJ8WmGG5b4VQIbcbJ20S+pFzePSUH//i5Eu/tqbDr9dKlN+KJT44hVSXF6ptHev143pIWK4PeyOLzYzV4ZMYgtwv20cIVeclQSITYfrLOLOikZqC/zY5SgcTtGTNnzhw8/vjjAEgKk1AoxMmTJ7F7924sXrwYy5cvh1qtxpEjRzB16lQwDIOMjAyYTCa0tLiu2gwmtrnO4SZGIrREfYnmaJ1hGPzvzSO9XgjylzHZ8fj80SuQmRCDu/91CB8fvID1v5IFmmevH+Z0k+lQwG/u7G5RFAiub24Lb7kEI4WMh18Y7TaYsPz6YV7/ncgcxfMN1wDyvtjOlPjURV8ai6WopLitIBtbjl6061G+5utSVDZr8OptYywZMYGEn2nFy8V48MpBAX/8SEQmFuKqoan4vqQeJnO2U/9EedA0yu2jKhQKKJVKqNVqPPbYY1iyZAlGjx6Np59+Ghs3bkR2djbeeustqNVqKJVKu7/r7AxMT+2+Du/LBrKTn7dkxsdg80OXY2peMpZtPYEXtpVgxpCUHmmdoSQ1ljQPi5RcYoUlQg9escqwfrFgGGDRJN88Y/6iluKmXUWcJUL37fv1wPRcsBxJHQSAnSX1+M+BC3hgWi6mDAp8sRsAZCcS6/NPV+VZLkSXArNHpKNJrcPRC62oaNTYtZQONB4vE7W1tbjrrrtw8803Y+7cuZg1axZGjiTTsVmzZqGkpARKpRIajbUARaPRQKWK3sUOX+AFPRg5zt6gkonxwR8KcPeUgRiQKMdf5o8OmrXgDbeMy8Tnj14R1IjYF/gZQzDHkx4nw2cPXY7nbhju0985pi06w+qh+yaQ2YlyzB3dDxt/PY+yBjWWbinCsH6xePLa4PnaozLj8MkDl9k14LoUuGpoKiRCAb4uqkVlsyZoC6KAB0FvamrCvffei6eeegrz588HANx3330oKioCAOzfvx8jRozA+PHjsXfvXrAsi5qaGrAsi8TE0KcJRiJWQQ9fRCISCvD8TSPww/+bYcmyCRdSkRCjs+LDOgZbLJZLkD+fCQMSfa60TFBIIBUJ3C4Kzxqehsevzvdrke2hGYOg0Zsw7+1f0Kkz4o2FYwOyr60rGIbB5NykoCUDRCpKqQjT8pPx2ZGL0BvZoC2IAh4WRd955x10dHTg7bffxttvvw0AeOaZZ/DSSy9BLBYjOTkZq1evhlKpREFBARYsWACWZbFy5cqgDbivkRRGy4XiGT4PPZC7FQWK+ROycMWgZIst5IwUlRRPzPIvqh6aHouZQ1Pxw6kGrJo73LJFIiXwzB6Zjl2nSGfTYOWgAx4E/bnnnsNzzz3X4/ZNmzb1uK2wsBCFhYWBG1mUMCBJgX5xsohZqKXYk6iQQCYWoL8Pe2SGCqlIiIFB9FsB4H9vHoGrhqRg8eTwratcClwzLA1CAQMTywXVcomMlako5sErc4NW4k/pPQqpCD89dVVI9pmNRLIS5Ljz8oHhHkbUk6iQYHJOIoprOiyz9mBABT3IyMTCXnWpowSfYBYuUSg8z980AtVt3UFNSqCCTqFQKCFgcJoq6OsU1NilUCiUKIEKOoVCoUQJVNApFAolSqCCTqFQKFECFXQKhUKJEqigUygUSpRABZ1CoVCiBCroFAqFEiVQQadQKJQogQo6hUKhRAlU0CkUCiVKoIJOoVAoUQIVdAqFQokSqKBTKBRKlEAFnUKhUKIEKugUCoUSJVBBp1AolCiBCjqFQqFECVTQKRQKJUqggk6hUChRAhV0CoVCiRKooFMoFEqUIHJ3p8FgwPLly1FdXQ29Xo+HH34YeXl5eOaZZ8AwDPLz87Fq1SoIBAKsW7cOu3fvhkgkwvLlyzF69OhQvQYKhUKhwIOgf/nll4iPj8df//pXtLW14ZZbbsHQoUOxZMkSTJ48GStXrsSuXbuQkZGBgwcPYvPmzaitrUVhYSG2bNkSqtdAoVAoFHgQ9Dlz5mD27NkAAI7jIBQKUVxcjEmTJgEApk+fjl9++QU5OTmYOnUqGIZBRkYGTCYTWlpakJiYGPxXQKFQKBQAHjx0hUIBpVIJtVqNxx57DEuWLAHHcWAYxnJ/Z2cn1Go1lEql3d91dnYGd+QUCoVCscPjomhtbS3uuusu3HzzzZg7dy4EAuufaDQaxMbGQqlUQqPR2N2uUqmCM2IK5VKkuRzguHCPghLhuBX0pqYm3HvvvXjqqacwf/58AMDw4cNx4MABAMCePXtQUFCA8ePHY+/evWBZFjU1NWBZltotFEqgaKkA3hwPVPwY7pFQIhy3Hvo777yDjo4OvP3223j77bcBAM8++yzWrFmDtWvXIjc3F7Nnz4ZQKERBQQEWLFgAlmWxcuXKkAyeQrkk6KwnP9uqwjsOSsTDcFz45nHz5s3D1q1bw/X0FErf4OxOYOPvgWueB6Y+Ee7RUCIAV9pJC4solEhHryY/u1vDOw5KxEMFnUKJdPTmhAMq6BQPUEGnUCIdXtC7WsI7DkrEQwWdQol0LJZLW1iHQYl8qKBTKJGOxXKhETrFPVTQKZRIh3roFC+hgk6hRDo0y4XiJVTQKZRIh4/QjVpA3xXesVAiGiroFEqko7f2SaJROsUdVNAplEiHCjrFS6igU/om3a3AV09cGql8ejUglpP/00wXihuooFP6Jsf+Axz+J3Bhf2if99ulwBtjgZNbQ9fOVq8B4rLI/2mETnEDFXRK3+TEZvJT0xja560tAlorgc/uAd6fCfzyd6DhVHCfU68B4rLJ/6mgU9xABT3QlHwBaDvCPYropqkMqPmN/D/Ugq7rBPJnA3P/TrJOvl8BvD0ZKP0qeM9pG6GHs/zfZARObqGZNhEMFfRA0lQGfHoXcPTf4R5JdHNiMwAGEEoATVNon1vXDsTEAxP+ADyyH3iiGEgcBOx9PTjPx3HEQ1ckA0Jp+CJ0bTvwn1uBz+4FjvwrPGOgeIQKeiCpPUZ+1peEdRhRDccRQR84FYjNCE+ELrXZXjEuC7jsYaD6MFB1sHePfew/QOs5+9uMOoAzARIlIE8Mz6Jo63ngg9lA5R5ArACqj4Z+DBSvoIIeSOqKyM8GKuhBo+Y3oKUcGHUroEgJraBzXE9BB4AxiwBZHPDr2/4/9ulvgc8fBva9aX87n7IoUQIxCaHP6rl4GPjH1UBnDXDHVmDQVVa7ixJxUEEPJHUnyM/G0wDLhm8cHAccfL/3EWMkcuIzYrUMv8ks6CG0XIxagDUC0lj726VKYPwfgJIv/dsmTq8Bvnma/P/iIYf7zGX/EgUQkxhay6X4v8CHN5Dnvm8nkHslkDGOXFC17aEbB8VrqKAHCo4jGRBiOWDsBtrOhW8sjaeBb/4f8M/ZwI8vkcUsf2k9B3TWBWxovYI1kUW5/GtJtKpIDm2EruskPx0jdACY9AD5uf8t3x93z6tA+wUgZzpQd9J+0dESoSuIdx+KRVGOA35+Ddh8N9BvDPDHXUDKYHJfxjjys/Z48MdB8ZnoEHSTAfh+JaBu8O74ox8B2x4Hfvor8S0rfgKaywFDt/9j6KwFupqAYXPJ7w2l/j9Wbzn9Dfk5bC7w0yvAv+YALZW+P05nHfDudJJ3/eNL9hWL4eDcXkBdB4yaT37nI/RQzYYsgh7b8774bGDs7cCh94HGM94/ZuNpYrOMuR24/E/EL+fXYgAnlkuQI3SjHvjiT8Cu/wVGzgfu+pJcOHl4Qae2S0QiCvcAAkL1UeCXN4iPOe1/3B/b3Wae3nJkCu2IPAmIzSSLXbGZQFwm0G8s8Q7dwdsto24Dij4hPvrQG/x4MQHg9LfkxLvtIxLRbnsCeGcqcOPrwOjbvH+cb58GDFogfxa5MBz9iGxUPOo2QBCGWODEZiJsg+eQ3xUpRAC1bWTB0Bt++TtZ3LvjM9+fX2dOR3UWoQPA1atI2ur2Z4A7tgAM4/7xOA74+n9I9D3rfwHG/J5WHQQGTCH/t7Vc5GbLheM8P7Y/dLWQLK1zPwNXPgPMeKbn88gTgfgBVNAjlOgQ9FZz9Fn+o2dBP7GZWCIP/ASkDAU6qsm/9mqg4yL52X6RWA3nfiFpamCApyvci0ateUE0exIQ3z98Ebq6gfiwVy0nv4/8PZA1Cdh6P7D1ASCrAEjM9fw4p74m4nT1SvKeXvgV2L4M+O+DwMH3gIUfA6q04L4WW4w64lEPmwuIY8htihTyU9PovaBX/gRU7Cb2jUDo2xjcWS4AoEwBZiwDdiwDzmwHhlzn/vGKPiXieePr5G8B8tnY+uh2lksCYNIBhi7yeyBpLgf+cxvQdgGY9777C3/GOCroEUp0WC68nVB1wHPRw9GPgPTRQMZYQCwDkgYR73LsImD6U8Dcv5Ho7ZH9wLILwIKNADig6az7x60rIiejLBZIHR4+QT+zHQBnLybx2cD8f5EI8LAXOcTadhI5po0EpjxGbut/GfFSf/cueW2fLCbRe6g4+z25uPJ2C2C1Anzx0dsukIXNzlrfx+BJ0AFg0v1A8hBiUbmjuw347lkgcwIw/m7r7VkTiaDzbQXsBN180Qq07XJ+H8lk6WoB7vrC8ywuYxwJeOgepxFHdAg6n7tr0pMvpytqjhHhHX+X94+dOoz8bPZC0NNHW/+m6QzxI20JhQd9+lsgrj8RY1ti+xEL6LcNnoV45wuAup5UQwrF1tsFAmDMQuCW/yOi89WS0PQz4ThSrCVPBnJmWG+3ROheZrpwHBF0wPrTF7wRdKGY2HOO+eSO/LAa6GoGblhrb19lTSTvfbs5W8ZiuZg9dCCwQlr5M/DRzcRq/ONOq9Xjjoyx5Ket10+JCLwS9OPHj+POO+8EAJSUlGDatGm48847ceedd+Kbb8gC3Lp16zB//nwsXLgQRUVFwRuxM1orgcwCUklX/oPr445+BIhkJIfZW+IHAAKx+whd205O4PRR5PfU4SQKbCm3HnPxCPDn/kD1Ee+f21f0XcR2GnKdc4914h9JYUrJ564f48KvwOEPgMkPAVkTnB8z4hZiLRz/uGfedDDY/Wfg7HfAlD8BQhuX0NZy8QZNo3XdxFbQiz/3TiTdLYraIk8mfrvjBZ2n+ghw6ANg4v1WceTJmkh+8raLo+UCBDZCP/xPQBYP3Pc9ma16Q78x5Gc4bZfi/wLqEBeV9QE8Cvr777+P5557DjqdDgBQXFyMe+65B+vXr8f69etx/fXXo7i4GAcPHsTmzZuxdu1avPDCC0EfuB2t54DUocCAy4GKH50fo+8i/vnwW0j6l7cIRcRKaS5zfUzdSfKT/6LzUb1tgVHNUSLyh/7p/XP7SsVusj7gyrvNmQ4k5QOH/uH8fqMO+LKQRPhXPev+uaY/Td7L71cCZ3b0ZtTuOboe+OnPwNjFwBVL7O+LSQTAeB+h24p463nzbVXA5j+Qi70nPC2K8vB+fldzz/tYE/DVk4AyFZjp5D1OGwGIYkhBD2AVdLHc+riBEnSTESjfBQy+1vs1CIBcWBJzwyfojadJSuW+N8Lz/BGMR0Hv378/3nzTGoWdPHkSu3fvxuLFi7F8+XKo1WocOXIEU6dOBcMwyMjIgMlkQktLiPw1vYZMURNygNyriIg6y5su+YKckL7YLTzJ+cRCcQVfIcpH6En5ACO099F5n794a/Cad53+hkSPA65wfj/DkCj94iFiPzlS/Dl5nTe8Sopl3CEQEOul32jgs/uCs2ag6wS+fhLInQHMfaPnrEMoIkLkbYRua4Pw4s6Pu+28d+MRiAGR1P1xvLfvTNAP/5NYFbNfIllZjgjFJGrny+v1alJuLxDYROgBOreqDpDZZf5s3/82Y5zz71AoKN1Gflb8FJ7nj2A8Cvrs2bMhElmnuaNHj8bTTz+NjRs3Ijs7G2+99RbUajWUSqsAKBQKdHZ2BmfEjvAnaWKONbWw3EmUfvQj0kTJG4/QkeR8IsiuCnRqiwBFKqBKJ7/zi612gl5BfFBDF3DSj5Q5T7AsWRDNnwWIJK6PG7OQRHuHP+h5Hy9yuTO8e06JnGS7SOTAxwsDv0jWdIasi0y8397Lt8WX8n/+9aWOsAp4o/kzavVS0KUqzymD8iTy01HQO+uBXauBnCtJ9pErEnNtPHSNNaPFX8uF40gWiyNntpMLlKeUXGdkjCNjDIftwQt63Qm6MOuAz4uis2bNwsiRIy3/LykpgVKphEZjXfDTaDRQqTxMSwMFL+gJA4G0UYCqH7DzeXsvvekscGEfic79yd9NygdYg+soru4EiVRtSR1mb7m0VBChTB3h3fTeV6qPEGEbcr3742LiyRpC0eaefUE0DSRq9BSB2hKXCSz8D9BRS3KYTQbnxxX/F9i3zrfiLX7dIjnf9TG+lP+3XSA2Tdpwmwj9lPU+Tzjr4+IMOR+hO4zru+eIJXbDWvffw7gskoVjMtgLujiG2DG+CnrlHuDN8T2bap39jgQ43rwmRywVo8d8/9ve0HaBPOeQGwBwJO2TYsFnQb/vvvssi5779+/HiBEjMH78eOzduxcsy6KmpgYsyyIx0QdPrjfwVkZCDpmWLt5MRGn974AdzxJf+OhHgEBEmij5Ay8ozhZGjToS5fF2C0/qcDI2QzfxTVsrSeQ1/i7iPfJ564Hi9NfkNeZd7fnYiX8kwnL8Y/vb1Q1kpuErWQXATW+Sk+ubp+wzX1iWRKWb7yZpeusmEmvHG5rOEusqIcf1Mb6U/7ddIDUC8f1J7YHJaI3Q26us4z7+CfDP63pm8Gg7PC+IAjYRuk30WLkHOPEpcMXjQHKe+7+PywI4FuioMQu6jf2lTCW3+wIvuhd+td7Weh5oPAUM9sNuAcwZXYxvPjrHkeSA3mRG8X3nr1lF3hdqu9jhs6A///zzeOmll3DnnXfi6NGjeOSRRzBy5EgUFBRgwYIFKCwsxMqVK4MxVue0niMCzi/qpI8CHthNRGv/OuD9q0l5/+A5/hfCJJlPQGepiw2lZLEz3UmEDo4s4HTUEOsgMdec48uQ9EJ3sCy5WDj95yR74vS3JNrip+Xu6DeaFBsd+of9yaVuIILhD2MWAFOfIL2y+UVXQzew5T7g51eBcXcCd35OZgib/+BdL5Dms0DCAPcWkq+WS8IAkrnEGomoN54mHrVRa20dceZbMqNzjNp1naTOwBMxCbBbrGVNJK8/foDnwjfAuplF+0XiodsWESUNsrdPOmqA71a479fDtyKosYnQz35HfvrjnwPkfUjO903Q64qAf8wkhWn+UrqNzHJThpDve+Ue/x/LGe3VJDHg8L/I+9/H8KpSNCsrC59++ikAYMSIEdi0aVOPYwoLC1FYWBjY0XlDayWxW2yRyIEbXgPyZgFfPEqmvv4shvLIE0nU5SxC50v++QwXntTh5GdDKenbDRBBlycSEeqodv18LEtK9RuKXR+TNhLIu4Z45opUEm1NuMf71zTxPlL1WfmT1TPXNPTMX/eFmSuJhfHtUiJqB94hC7DXvEAiU4YBMr4C1g4Hfv0/4HfvuH+8pjIgebD7YxQppPTfqHcv/BxHovAhc0iEDgDnfyFrGkNuIDOctgvkos/bMLXHyQWAR9dh/SzdIRSZG2mZPfTWc2Q94MbXrVWu7uC3m2u/SCJ028XTpDzg4iZr+f+Jz4B9fydrI2kjnD9e02ny09ZyObODfB89zRbckTHON0HlBfKnV8wth724ONqibiB7yF65lPyeM51cmDpqvPtcvIFvccHboilDredZ/8t9syPDQN8vLGqpdD0lHzIHeHgf8PsPSIe+3pCU70LQi8jUz3EMCTkkL76hhPjngLXkXpXuvoNhzW9EzMcsIqX3jv+uNAvm/nWkvenbl1lfr7cMv4WM23brNHWj/xE6QCyv379Poqct95F0ztvWA1OXWD3jmHhg3GIiRO7eA5YlefxJHgTHXUaJLeoGEoXHD7AKOp9umT+L/Gw7T3xrPkXVcRbhrYcOkACA99D5i7c3LRcA0kMIIBcgxzL/xEHkwsLPShrNFx91vfPH4jgSoQtE5P3sbiMpvOd+9j8658kYR7z+Di+rbvkxdzX7V79wfBMAjrROBsjiMhA426WznjzHhHuARw4A175IztWD75Hiq1dySEZXpHQfdULf7uXCmkhUxX/AzlCl2ZeL+0tyvrms3oHaIhLVOjarEopIy1E+00UotZ6oqn7uS89LvyQn4JyX3Vso2g4SIZV9T2wDx5mKO8QyctHhsykMWlJa74+HbotUBSz6mCxMX17ovDhp8kOkX/uhD5znYgNkXEat+wVRwLoAqWkk1bCu4Be04/ubLQ3GunDOX+zbLhA7gzUv7PZK0JOtF5l2s6DHZnn3txI5uSBYLBcbD91i/5WTiy///XLVaVRdTz5XfhZS85vZttOS/PPeYLsw6u695+EFfeiNJBiZ+EfvbVCjnszqBk6zzkTSRpJF7so9pHVHbzn4LrFGpxQSayt1KClm06nJBfDsd8Cxj0nu/o2vAyN+1/vnDDB9O0LvqCYnny9C5i/J+eQLaZsZwrJA/cmeGS48fE+XlgoyRl70VWnuI6rSbeSL68kPl8UCw24kOdpzPPQOcUZclnUazJ9svYnQeRIGArd+6LrSNGkQWdM4/IHrrBd+NpTkQdC9rRbl/fD4/mTarOpHIl1lOsnUkSeTY/iIN20kESrbNQZfI3SNWdA7zO+xL7YA/9nYZrkAQJI5ym8uI9+/RrOd4ipq5O/n+7PUHAXO7iABgKt6BW9JH0X6A3nro2uaAGkc6Sxp0pOCMW85+RnZNcm2uEwgAHKmEduwty0odGoSYAy7sWfFrFRJivVufB146GcSCG2+G9hyf/j2eHVB3xZ02wyXYMMLi23FaGsliaAcM1x4UoaSk7n2uP10W9WPRFTOFrIaT5GpMd9XPZjEZVkjdI05wguEoHvD5Y+SCHb975wLAr8A7SlC97afCx+h8/40b7ukDrX+3mbO/ABDZnWaRqtQGnWk06G3gq5Iso/QYxJJ5O0tcdnOBT2uP8kdbyk3WzLmdGFXETpfEJc9idg11UeBM9+R3PPe+sESBWlE5rWgNxKLLGkQsTWO/Jusk3iC40jb49QRPbO4cq4kgZ2zPHtf+G09WYuZ8rj745LzSZuEGctJa+q3pzivewkTfVvQbXPQg42z1EV+Su6Y4cLDL4y2VzkIejoAziqitpRuA8CEppd6XBapFNR2WAtEemu5eEvONOCmdeT9fO8q4PNH7L3YprMkmuMF2xXedlxsu0CicL4Cll/sTDG3aYjvT45pKCXfp/6Xk9v5z1hnbpLlTdoiYPbQm4kYdVSTWYAvxGWRC4xRa2+5CEVkfM1l1ugbcD3jazwNSFQkiMgcD5TtJEFGb9eUePiKUW8iZE2j9fO8cilZIN7lRZuQs9+T9NIrHuuZv88v6Ff2wkc3GchOU/2nANkTPR8vFAEzlpJmZlIlsP4WsseCp06vIaCPC3oliVbivPQme0PCQOJr26Yu1p0gt/G9WxyxvT3RZhahNFeUOpsml34JZE+2Vp0GE/5966i2idA9CGggGX8n8NhRcqKe2Ay8OQHYb95oufksycDwVAgmiyPFNvzCsyv4HHQepxF6FRH01GHmbB/GRtC97OPCI08mdqCug0To3vrnPHFZZEEU6Nn7PCkPaK6w5tCnDHUt6E2nSTDCMEDGeGtzskAKuqbBu9x4TZP1AqxMIV516ZdA1SH3f7fv72T9yVl1bWIuua83gl78OQm6rnjMt7/LHA88uAeY/DDx39+d7l3zve424Nd3grLTVh8X9HPkRPR1owJ/EIqJqNv2dKk/SaacrqaucdnW6KpHhI6egt5SSS4SobBb+PEBZGrPT9lDFaHzyMye6qMHgYFXkM0hTm/3LmURIEKVfw2Z2bAm18e5EnQ+Qk8YQCyVptNEIKVKIoQWQfeida4ttuX/HRf9i9B5egj6IGK5NJSS4CBliJsI/Qy5HyACBJAZpTeLmN7gy5Z0vOXCc/mfSMS+c5XrCL/6CFmQvOxh5+0fGIbYLpU/+yeQHEeafCUP8S/rRxwDXPdn0kfe0AX8Yxbw48uuK6YBsrvajmXWC3YA6duC3uIkBz2YJA+29/zqS0gZuSsEAiIOQE8PHeiZ6XLKnEI47Mbej9UbLAUsVeRkk8aS7JdwkJgDLNhA2jd88QhZAPOUssgz6lYSJbrKiWZZEn3bCvrwm4FZq0mVK0DSGXn4mVW/Mf4LOi9creeJrRXrq6BnW//vTNCNWuLdpg4FlC4W2bXtZA9W/sKYPpoEGMPcZIX5SvpIUs3rSdBZE7m42VpoUiWxXs7/Yi10cuSXvxPrbfwfXD927pWkYVn9Sd/HX/EjCaKmFPZuW8XcGSRFetR8stj7wSzne8uajMCxjeTi4akBnh/0bUFvrbS3MoJNUh6Z2rMmMm3quGj1yV2RNtxsC9mcoIoUAEzPCL10G1lgDdVFSpVOTkY+QvfkVwcbkZTksfN+tacFUZ78a8nF6ISLpmeaBhJ92wq6LI5MsfnZne19fETbbwz5jDVNfkTo5splvhOn7efvDXYRusOJn2jOwlDXkRmGMpWIt+PGJfx6D/96JHLgT4dJXUCgEMeQc8CToHe3knYGjt+xCXeTYGfn8z1nWC0VxJIpuMd9EVLOdPLTH9vll7+TWY4ve+26IiYemPcecOu/yYX83Wk9rZWzO8jFtzeFjm7ou4Le3Uq+xKHIcOFJzifCwC+eAa6r83imPkk2a7bdmEEoIieh2kbQO+vI5sCBjJ48IRCSyLH9IonQQ5Xh4o7UYcC1a0g6nKvFZkfEMcSmKv3S+W5MfCdF2yjcEV5wGYE1ouWrf2uPe7+5BQ+fH8/37PHVclGkkkAAcB6h86QOta7J8FE6y5KLIn8xSR5iPT62n+vOlf6SMZYIuruFUX7R2tZyAchYrl5JCvCOO1Sg73+LBByTH3L//LEZJAvN1wKj2uMkQr/socBWgI64hWxhmTMd2L4U2PA7a3rw0Y/I5xWoNQwH+q6gW1IWB4buOW1TF/myfE8RemIOMNRJB0THatFTXwPgQuef8/D5zpEQofNMfgB4qty32deo+WQB0tnUnc9BT3Aj6BI5ef0JA63l+fwFpfa4H4uiSda/BXy3XAQC60XAUdBVGWQhGDBH6ObiHH4dZON84OVM4KsnSEFbsM+RjHHE8nDXsdIi6E6+Y8NvIXur/viitS5B0wz8tpH0CPLG78+dQQrF1v+OLKw3nfWcebPvTTL78aVlhreo0oHbPwVu/BtZ9H17CimMOvsdMPZ2+wAvgPRdQW81C3ooLRfb1MX6EuLt+Ztho0y399BLtxFLh/fcQwWfi67pRWOuYODLDjoAMHA6iWpPbO55n2MOuisyJ9gX28TEEzG0E3QvI3SJgohpcxkAxr9eI/x4HS0XgcC6JpMyxPq5qevJYty5n8lmL7NWk3WJIImHBW8WRt0JOsOQfj8d1dbGXYfeJx1B+U3KPXHVchLJt18kC47rCoA3xpCmaKe/7bmfb9sF4ORWYvn4soOZLzAMsYse3ktmUtufIbbTuDuC83zoy6X/ocxB55Enkf0Xm86QApTUYf71VwfIFZw/AbpayEk4pdD/x/OXuCzgZDXAmUKf4RJIhCJg5DzSJU/bbt/Qqu0CERJPhT0LP+55G78wmjSITP+9aa4FkM9RkUxESpnun83BBwuOETpAhFzXScTIyFsudSTYMOnJln2jfdg7tzekjSD2UM1vxG5wBl8162oWmDON2BA/v0YWuQ+8S6qJU4Y4P94ReaK5WvolYrGVfQ+U7SKl+of+AQglpDtj3izSu+fwv8hndNnDvr5a30nMBe75lkToerX3e7f6Qd8V9JZKIkDOvuzBgmFIlN5cRiL0UW52nfGEqh+JWkwG0iSKNYbebgHMvbfNi1GhzEEPBqNuJR0eS78iDcB4HFMWXeEsy6HfGLJ9YftF73YrskWe6F9REY9F0J1ciK5dQyobAbNfzxDLhe/+md6Lrpm+IpKSxX9PETojcN/O4upVpMvohzcSC+cKD1WbrkgYQPrETPwjqfC9sJ8UJ5XtJD35vzP3DxqzKDQ1LABZr5ryp6A/Td8V9NZzobVbeJIHk5Jfo9azf+4OVRoAjpyEpduIx5oxPmDD9Bq77Js+HKEDxDJJGEhsF0dBd9VvxxP8wuj5fd7bLTz8wqiv/jnPqNuICMrie94Xl2m9UAhFZDagrie5zUKp5x44gSZjHHDyv9a2vo5oGskM113NSPpI0gb4+MdAZoG1Wrc3iKTEX8+dAcx+kaSvlu0km3Bf+VTvHz/C6MMe+rnQ2i08SXnWajtPGS7u4HPRW8pJ97ahN4bebgHsI5RI8tD9gWFIlF75E2mFCpCMj/Yq7yJ0Z6SbBb29yvet2viFUX+jwJTBxBv25nuhTLdG6KnDgu+bO5IxjnR1/PAGYNvjJB2w9Cur1WJb9u+Oq54l2UhXLQvO+RCfTXztW94Kj34Emb4p6EYdmQKHMmWRxzY32lXJvzfw1aK/bSQXiHDYLUB0CTpABJ1jyR6mAPGVTXr3KYvuUKZYI2xfBV3RywjdF5SpJGuq7mRo7RaeoXOBsXeQ97rkS+D7FcAni8k+s4B92b874rOBJUVkUwmKz/RNy6WtCgAXHsuFn8rGZnq33Zsr+Nzh4q0kkgvE9NIfZLEkWycQvdAjgZQhpDjrxGaSX2xpm+unoAPmAqPqXkTooRD0NOIVG7pItW2oUSSRqJenuxX44UXg8D9J8zdNI8lXpwSVvhmh8ymL4ZgyJeYQX7M3/jlApp+MgEQ0Q64P/RTZlrgskhrnS3vXSGbUrUD1YVJpaNsH3V94H91fQfe1MZc/qNKsvUFctXMOJTEJpL0CZyIZXJqmyKlziGL6pqCHsg+6IyIpKUQYs7B3jyMUWSPiUFaHOiO+v7U4JRrgu/Kd2GKzU5GPpfe2+CvoA6cR68Bdv59AYfv59WZtJ5BkTwLEcrLTl67dO8uF0iv6puXSeo58UcLl+d64NjCPo0ojBQ+5Vwbm8fzlmlURt/NKr4jLIgVCJz4lrYgVqd7njzvDX0FPGQzcscX/5/UF/lyI6x+8QhlfEUnJ51DyBfmdRuhBp28KuqbBXOgRhqyQQDL+LrLAG+6dxHuzuBupjLoV+GoJKb7pjd0CkIykiX8k25BFKvyaTCTYLbYMmkmKfAAq6CGgbwr6jGVECPs6E/8Y7hFEL8NvBr55irRXGDCld4/FMMANrwVmXMGCt1zCkeHijkFXWf9PBT3o9E0PPWlQaHxJSt9FnmhNfetthN4XSBhIAoTRC8I9EntShlprLqiHHnS8EvTjx4/jzjvvBACcP38eixYtwu23345Vq1aBNff6XbduHebPn4+FCxeiqKgoeCOmULxl1Hzy81IQdKGIzCKC2CfELxiGNAoDrJWzlKDh0XJ5//338eWXXyImhiwqvfzyy1iyZAkmT56MlStXYteuXcjIyMDBgwexefNm1NbWorCwEFu2hGgxiEJxxdAbScOzISHYcJvimimFQFKu+00qKAHBY4Tev39/vPnmm5bfi4uLMWnSJADA9OnTsW/fPhw5cgRTp04FwzDIyMiAyWRCS0tL8EZNoXiDWEaaWKmiKCWzL5I2HJgefX1TIhGPgj579myIRNZAnuM4MObsEoVCgc7OTqjVaiiV1p7N/O0UCoVCCR0+L4oKbFqMajQaxMbGQqlUQqPR2N2uUvmYs0uhUCiUXuGzoA8fPhwHDhwAAOzZswcFBQUYP3489u7dC5ZlUVNTA5ZlkZjo444zFAqFQukVPuehL126FCtWrMDatWuRm5uL2bNnQygUoqCgAAsWLADLsli5cmUwxkqhUCgUNzAc52kn1eAxb948bN26NVxPT6FQKH0SV9rZNwuLKBQKhdIDKugUCoUSJVBBp1AolCghrM25qqurMW/evHAOgUKhUPoc1dXVTm8P66IohUKhUAIHtVwoFAolSqCCTqFQKFECFXQKhUKJEqigUygUSpRABZ1CoVCiBCroFAqFEiWEPA/9+PHjePXVV7F+/Xo0NzfjueeeQ0dHB0wmE/7yl7+gf//++PTTT7Fp0yaIRCI8/PDDuOqqqzw/cAjHXVpailWrVkEoFGLgwIF48cUXIRAIImrcBoMBy5cvR3V1NfR6PR5++GHk5eXhmWeeAcMwyM/Px6pVqyAQCLBu3Trs3r0bIpEIy5cvx+jRoyNmzBkZGVi9ejWEQiEkEgleeeUVJCcnR8x77WzMV199NQBg27Zt2LBhAz755BMAiJgxuxr32LFjI/p8dPX9iPRz0WQy4bnnnkNlZSUYhsELL7wAqVQanHORCyHvvfced+ONN3K33norx3Ect3TpUu7rr7/mOI7j9u/fz/34449cQ0MDd+ONN3I6nY7r6Oiw/D+cOI77kUce4Xbv3s1xHMc9+eST3K5duyJu3J999hm3Zs0ajuM4rrW1lbvyyiu5Bx98kPv11185juO4FStWcN999x138uRJ7s477+RYluWqq6u5efPmRdSYFy9ezJWUlHAcx3Eff/wx99JLL0XUe+1szBzHccXFxdxdd91l+c5E0pg5zvm4I/18dDbmvnAufv/999wzzzzDcRzH/frrr9xDDz0UtHMxpJaL43Z2R48eRX19Pe6++25s27YNkyZNQlFREcaNGweJRAKVSoX+/fvj1KlToRxmDxzHPWzYMLS1tYHjOGg0GohEoogb95w5c/D4448DILtMCYXCiN8+0NmY165di2HDhgEgkY5UKo2o99rZmFtbW7F27VosX77cclwkjRlwPu5IPx+djbkvnIvXXHMNVq9eDQCoqalBbGxs0M7FkAq643Z21dXViI2NxYcffoh+/frh/fffh1qtttvtSKFQQK1Wh3KYPXAcNz+1u+6669Dc3IzJkydH3LgVCgWUSiXUajUee+wxLFmyJOK3D3Q25tTUVADk4r9hwwbcfffdEfVeO4758ccfx7PPPotly5ZBoVBYjoukMfPP7/heR/r56GzMfeFcBACRSISlS5di9erVmDt3btDOxbAuisbHx2PmzJkAgJkzZ+LkyZN9Yju7F198ERs3bsT27dtxyy234M9//nNEjru2thZ33XUXbr75ZsydO7dPbB/oOGYA+Oabb7Bq1Sq89957SExMjOgxDxw4EOfPn8fzzz+PJ598EmVlZXjxxRcjbsxAz/e6L5yPjmPuK+ciALzyyivYsWMHVqxYAZ1OZ7k9kOdiWAV9woQJ+OmnnwAAhw4dQl5eHkaPHo0jR45Ap9Ohs7MT5eXlGDx4cDiH2YO4uDjLlTQ1NRUdHR0RN+6mpibce++9eOqppzB//nwAkb99oLMxf/HFF9iwYQPWr1+P7OxsAIio99pxzKNHj8bXX3+N9evXY+3atcjLy8Ozzz4bUWN2Nm4g8s9HZ2PuC+fi559/jnfffRcAEBMTA4ZhMHLkyKCciyFvznXx4kU8+eST+PTTT1FdXY3nnnsO3d3dUCqVeO211xAXF4dPP/0Un3zyCTiOw4MPPojZs2eHcogex3348GG8+uqrEIlEEIvFWL16NbKysiJq3GvWrMG3336L3Nxcy23PPvss1qxZA4PBgNzcXKxZswZCoRBvvvkm9uzZA5ZlsWzZMhQUFETEmE0mE86ePYuMjAzExsYCACZOnIjHHnssYt5rZ+/z+++/D5lMZvedARAxYwacj/vPf/5zRJ+Pzsb8+OOPR/y52NXVhWXLlqGpqQlGoxH3338/Bg0ahBUrVgT8XKTdFikUCiVKoIVFFAqFEiVQQadQKJQogQo6hUKhRAlU0CkUCiVKoIJOoVAoUQIVdAqFQokSqKBTKBRKlPD/AS6EXMv7efG9AAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 style="text-align:center;">Question # 3</h1><h4 style="text-align:center;"> Find out the range of age and frequency of heart attacks</h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[15]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">m1</span> <span class="o">=</span> <span class="nb">round</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">age</span><span class="o">.</span><span class="n">max</span><span class="p">()</span><span class="o">/</span><span class="mi">5</span><span class="p">)</span><span class="o">*</span><span class="mi">5</span>
<span class="n">m2</span> <span class="o">=</span> <span class="nb">round</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">age</span><span class="o">.</span><span class="n">min</span><span class="p">()</span><span class="o">/</span><span class="mi">5</span><span class="p">)</span><span class="o">*</span><span class="mi">5</span>
<span class="n">L</span><span class="o">=</span><span class="p">[</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">m2</span><span class="p">,</span><span class="n">m1</span><span class="p">,</span><span class="mi">5</span><span class="p">)]</span>
<span class="n">dicts</span><span class="o">=</span><span class="p">{}</span>
<span class="n">M</span><span class="o">=</span><span class="p">[]</span>
<span class="k">for</span> <span class="n">a</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">L</span><span class="p">)):</span>
    <span class="n">dicts</span><span class="p">[</span><span class="n">L</span><span class="p">[</span><span class="n">a</span><span class="p">]]</span><span class="o">=</span><span class="mi">0</span>
<span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="n">df</span><span class="o">.</span><span class="n">age</span><span class="p">:</span>
    <span class="k">for</span> <span class="n">k</span> <span class="ow">in</span> <span class="n">L</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">j</span><span class="o">&lt;</span><span class="n">k</span><span class="p">:</span>
            <span class="n">dicts</span><span class="p">[</span><span class="n">k</span><span class="p">]</span><span class="o">+=</span><span class="mi">1</span>
            <span class="k">break</span>
<span class="k">for</span> <span class="n">b</span> <span class="ow">in</span> <span class="n">dicts</span><span class="p">:</span>
    <span class="n">M</span><span class="o">.</span><span class="n">append</span><span class="p">(([</span><span class="n">b</span><span class="o">-</span><span class="mi">5</span><span class="p">,</span><span class="n">b</span><span class="p">],</span><span class="n">dicts</span><span class="p">[</span><span class="n">b</span><span class="p">]))</span>
<span class="n">M</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[15]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>[([25, 30], 1),
 ([30, 35], 2),
 ([35, 40], 12),
 ([40, 45], 40),
 ([45, 50], 32),
 ([50, 55], 56),
 ([55, 60], 69),
 ([60, 65], 49),
 ([65, 70], 31)]</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[16]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">s</span> <span class="o">=</span> <span class="mi">0</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">df</span><span class="o">.</span><span class="n">age</span><span class="p">:</span>
    <span class="k">if</span> <span class="mi">40</span><span class="o">&lt;=</span> <span class="n">i</span><span class="p">:</span>
        <span class="n">s</span><span class="o">+=</span><span class="mi">1</span>
<span class="n">m</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">df</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Most of people approximatly &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="mi">100</span><span class="o">*</span><span class="n">s</span><span class="o">/</span><span class="n">m</span><span class="p">)</span> <span class="o">+</span><span class="s2">&quot; who have a heart attack are over the age of 40&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Most of people approximatly 95.03311258278146 who have a heart attack are over the age of 40
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 style="text-align:center;">Question # 4</h1><h4 style="text-align:center;"> Which gender (M/F) most often gets a heart attack and the
percentage (%)?</h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[17]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">))</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">&#39;sex&#39;</span><span class="p">],</span> <span class="n">hue</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">&#39;output&#39;</span><span class="p">])</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xticklabels</span><span class="p">([</span><span class="s1">&#39;Female&#39;</span><span class="p">,</span> <span class="s1">&#39;Male&#39;</span><span class="p">])</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Heart Attack Frequency for Gender&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span><span class="n">title</span> <span class="o">=</span> <span class="s1">&#39;Output&#39;</span><span class="p">,</span> <span class="n">loc</span> <span class="o">=</span> <span class="s1">&#39;upper left&#39;</span><span class="p">,</span> <span class="n">labels</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;No Heart Attack&#39;</span><span class="p">,</span> <span class="s1">&#39;Have Heart Attack&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Gender&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>

<span class="n">hdf</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">]</span>
<span class="n">val_counts</span> <span class="o">=</span> <span class="n">hdf</span><span class="p">[</span><span class="s2">&quot;output&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">no_heart_attack</span> <span class="o">=</span> <span class="p">(</span><span class="n">val_counts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">/</span> <span class="n">hdf</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">*</span> <span class="mi">100</span>
<span class="n">heart_attack</span> <span class="o">=</span> <span class="p">(</span><span class="n">val_counts</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">/</span> <span class="n">hdf</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">*</span> <span class="mi">100</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Heart Attack to males: </span><span class="si">{</span><span class="n">math</span><span class="o">.</span><span class="n">floor</span><span class="p">(</span><span class="n">heart_attack</span><span class="p">)</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;No Heart Attack to males: </span><span class="si">{</span><span class="n">math</span><span class="o">.</span><span class="n">ceil</span><span class="p">(</span><span class="n">no_heart_attack</span><span class="p">)</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>

<span class="n">hdf</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span>
<span class="n">val_counts</span> <span class="o">=</span> <span class="n">hdf</span><span class="p">[</span><span class="s2">&quot;output&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">no_heart_attack</span> <span class="o">=</span> <span class="p">(</span><span class="n">val_counts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">/</span> <span class="n">hdf</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">*</span> <span class="mi">100</span>
<span class="n">heart_attack</span> <span class="o">=</span> <span class="p">(</span><span class="n">val_counts</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">/</span> <span class="n">hdf</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">*</span> <span class="mi">100</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Heart Attack to females: </span><span class="si">{</span><span class="n">math</span><span class="o">.</span><span class="n">floor</span><span class="p">(</span><span class="n">heart_attack</span><span class="p">)</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;No Heart Attack to females: </span><span class="si">{</span><span class="n">math</span><span class="o">.</span><span class="n">ceil</span><span class="p">(</span><span class="n">no_heart_attack</span><span class="p">)</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Heart Attack to males: 44%
No Heart Attack to males: 56%
Heart Attack to females: 75%
No Heart Attack to females: 25%
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 style="text-align:center;">Question # 5</h1><h4 style="text-align:center;"> Relation between cholestrol and age </h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[18]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df2</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;output&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span> <span class="p">]</span>
<span class="n">column_2</span> <span class="o">=</span> <span class="n">df2</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span>
<span class="n">column_1</span> <span class="o">=</span> <span class="n">df2</span><span class="p">[</span><span class="s2">&quot;chol&quot;</span><span class="p">]</span>
<span class="n">correlation</span> <span class="o">=</span> <span class="n">column_1</span><span class="o">.</span><span class="n">corr</span><span class="p">(</span><span class="n">column_2</span><span class="p">)</span>
<span class="n">correlation</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[18]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>0.24842077102722618</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 style="text-align:center;">Question # 6</h1><h4 style="text-align:center;"> Determining the range of vitals above and below normal level </h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[19]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">a</span> <span class="o">=</span> <span class="n">df</span><span class="p">[(</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;chol&#39;</span><span class="p">]</span><span class="o">&gt;</span><span class="mi">239</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;trtbps&#39;</span><span class="p">]</span> <span class="o">&gt;</span> <span class="mi">139</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;fbs&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">1</span><span class="p">)]</span>

<span class="n">val_counts</span> <span class="o">=</span> <span class="n">a</span><span class="p">[</span><span class="s2">&quot;output&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">no_heart_attack</span> <span class="o">=</span> <span class="p">(</span><span class="n">val_counts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">/</span> <span class="n">a</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">*</span> <span class="mi">100</span>
<span class="n">heart_attack</span> <span class="o">=</span> <span class="p">(</span><span class="n">val_counts</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">/</span> <span class="n">a</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">*</span> <span class="mi">100</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Heart Attack: </span><span class="si">{</span><span class="n">math</span><span class="o">.</span><span class="n">floor</span><span class="p">(</span><span class="n">heart_attack</span><span class="p">)</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;No Heart Attack: </span><span class="si">{</span><span class="n">math</span><span class="o">.</span><span class="n">ceil</span><span class="p">(</span><span class="n">no_heart_attack</span><span class="p">)</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>


<span class="n">sns</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;No Heart Attack&quot;</span><span class="p">,</span> <span class="s2">&quot;Heart Attack&quot;</span><span class="p">],</span> <span class="n">y</span> <span class="o">=</span> <span class="p">[</span><span class="n">no_heart_attack</span><span class="p">,</span> <span class="n">heart_attack</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Heart Attack: 57%
No Heart Attack: 43%
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 style="text-align:center;">Training</h1>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[20]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">RandomForestClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span><span class="p">,</span> <span class="n">cross_val_score</span><span class="p">,</span> <span class="n">cross_val_predict</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">metrics</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">datasets</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="kn">import</span> <span class="n">LogisticRegression</span>
<span class="kn">from</span> <span class="nn">sklearn.tree</span> <span class="kn">import</span> <span class="n">DecisionTreeClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">tree</span>
<span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[21]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;output&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;output&quot;</span><span class="p">]</span>
<span class="n">x_train</span><span class="p">,</span><span class="n">x_test</span><span class="p">,</span><span class="n">y_train</span><span class="p">,</span><span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">x</span><span class="p">,</span><span class="n">y</span><span class="p">,</span><span class="n">test_size</span><span class="o">=</span><span class="mf">0.3</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 style="text-align:center;"> Decision Tree </h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[22]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">decTree</span> <span class="o">=</span> <span class="n">DecisionTreeClassifier</span><span class="p">(</span><span class="n">max_depth</span><span class="o">=</span><span class="mi">6</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="n">decTree</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">)</span>
<span class="n">y_pred_decTree</span> <span class="o">=</span> <span class="n">decTree</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Accuracy of Decision Trees = &quot;</span> <span class="p">,</span> <span class="n">metrics</span><span class="o">.</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred_decTree</span><span class="p">)</span><span class="o">*</span><span class="mi">100</span><span class="p">,</span><span class="s2">&quot;%&quot;</span><span class="p">)</span>

<span class="c1">#Remove features which have low correlation with output (fbs, trtbps, chol)</span>

<span class="n">x_train_dt</span> <span class="o">=</span> <span class="n">x_train</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;fbs&quot;</span><span class="p">,</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_train_dt</span> <span class="o">=</span> <span class="n">x_train_dt</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;trtbps&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_train_dt</span> <span class="o">=</span> <span class="n">x_train_dt</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;chol&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_train_dt</span> <span class="o">=</span> <span class="n">x_train_dt</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;age&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_train_dt</span> <span class="o">=</span> <span class="n">x_train_dt</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;sex&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_test_dt</span> <span class="o">=</span> <span class="n">x_test</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;fbs&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_test_dt</span> <span class="o">=</span> <span class="n">x_test_dt</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;trtbps&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_test_dt</span> <span class="o">=</span> <span class="n">x_test_dt</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;chol&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_test_dt</span> <span class="o">=</span> <span class="n">x_test_dt</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;age&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">x_test_dt</span> <span class="o">=</span> <span class="n">x_test_dt</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s2">&quot;sex&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="n">decTree1</span> <span class="o">=</span> <span class="n">DecisionTreeClassifier</span><span class="p">(</span><span class="n">max_depth</span><span class="o">=</span><span class="mi">6</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="n">decTree1</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train_dt</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">y_pred_dt1</span> <span class="o">=</span> <span class="n">decTree1</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test_dt</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Accuracy of decision Tree after removing features = &quot;</span><span class="p">,</span> <span class="n">metrics</span><span class="o">.</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred_dt1</span><span class="p">)</span><span class="o">*</span><span class="mi">100</span><span class="p">,</span><span class="s2">&quot;%&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Accuracy of Decision Trees =  74.72527472527473 %
Accuracy of decision Tree after removing features =  79.12087912087912 %
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 style="text-align:center;"> KNN </h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[23]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#K Neighbours Classifier</span>
<span class="n">knc</span> <span class="o">=</span>  <span class="n">KNeighborsClassifier</span><span class="p">()</span>
<span class="n">knc</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">)</span>
<span class="n">y_pred_knc</span> <span class="o">=</span> <span class="n">knc</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Accuracy of K-Neighbours classifier = &quot;</span><span class="p">,</span> <span class="n">metrics</span><span class="o">.</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred_knc</span><span class="p">)</span><span class="o">*</span><span class="mi">100</span><span class="p">,</span><span class="s2">&quot;%&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Accuracy of K-Neighbours classifier =  71.42857142857143 %
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span> 
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span> 
</pre></div>

     </div>
</div>
</div>
</div>

</div>
</body>







</html>
