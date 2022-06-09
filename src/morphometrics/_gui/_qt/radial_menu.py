from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import napari
from qtpy import QtCore
from qtpy.QtCore import QLineF, QPoint, QRect, Qt
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QGraphicsOpacityEffect, QPushButton, QWidget

transparent = QColor(255, 255, 255, 0)


def getWidgetCenterPos(widget):
    """Get the center position of the given widget.
    Args:n
        widget (QtWidgets.QWidget): The widget to dertemine the center position
    Returns:
        QtCore.QPoint: The relative center position of the widget
    """
    return QPoint(
        (widget.rect().width() - widget.rect().x()) / 2,
        (widget.rect().height() - widget.rect().y()) / 2,
    )


@dataclass
class ButtonSpecification:
    name: str
    onClick: Callable[[napari.viewer.Viewer], None]
    onHoverTrue: Optional[Callable[[napari.viewer.Viewer], None]] = None
    onHoverFalse: Optional[Callable[[napari.viewer.Viewer], None]] = None


class RadialMenuButton(QPushButton):
    def __init__(
        self,
        name,
        parent=None,
        hoverCallback: Optional[Callable[[None], None]] = None,
        unHoverCallback: Optional[Callable[[None], None]] = None,
    ):
        super().__init__(name, parent=parent)
        self.setMouseTracking(True)
        self.setStyleSheet(
            """
        QPushButton
        {
            color: white;
            background-color: #232323;
            border-color: #232323;
            outline: none;
            border-style: solid;
            padding-top: 5px;
            padding-bottom: 5px;
            padding-left: 15px;
            padding-right: 15px;
            border-style: solid;
            border-width:1px;
            border-radius:4px;
            min-width: 10px;
            min-height: 10px;
        }
        QPushButton[hover=true]
        {
            background-color: #585858;
            border-color: #585858;
        }
        QPushButton:pressed, QPushButton[pressed=true]
        {
            background-color: #5479b5;
            border-color: #5479b5;
        }

        """
        )

        self._hoverEnabled = False
        self._pressEnabled = False
        self.opacityEffect = QGraphicsOpacityEffect(self, opacity=1.0)
        self.setGraphicsEffect(self.opacityEffect)
        self.setEnabled(False)
        self.targetPos = self.pos()

        self._hoverCallback = hoverCallback
        self._unHoverCallback = unHoverCallback

    def mouseMoveEvent(self, event):
        """Override the mouseMoveEvent method to avoid the event catch by the current widget
        Send the same event to the parent widget to keep mouse tracking on the RadialMenu widget
        Args:
            event (PySide2.QtGui.QMouseEvent): QMouseEvent sent by QT Framework
        """
        super().mouseMoveEvent(event)
        self.parent().mouseMoveEvent(event)

    def setHover(self, value):
        if self.isHovered() != value:
            self._hoverEnabled = value
            self.setProperty("hover", value)
            self.style().unpolish(self)
            self.style().polish(self)

            # if self.isHovered() is True:
            #     print("hover")
            # else:
            #     print("not hover")

    def isHovered(self):
        return self._hoverEnabled

    def setPress(self, value):
        if self.isPressed() != value:
            self._pressEnabled = value
            self.setProperty("pressed", value)
            self.style().unpolish(self)
            self.style().polish(self)

    def isPressed(self):
        return self._pressEnabled

    def animate(self, startPos, endPos, start=True, duration=100):
        self.parallelAnim = QtCore.QParallelAnimationGroup()

        self.posAnim = QtCore.QPropertyAnimation(self, b"pos")
        self.posAnim.setDuration(duration)
        self.posAnim.setStartValue(startPos)
        self.posAnim.setEndValue(endPos)

        self.opacityAnim = QtCore.QPropertyAnimation(self.opacityEffect, b"opacity")
        self.opacityAnim.setDuration(duration)
        self.opacityAnim.setStartValue(0)
        self.opacityAnim.setEndValue(1)

        self.parallelAnim.addAnimation(self.posAnim)
        self.parallelAnim.addAnimation(self.opacityAnim)
        if start:
            self.parallelAnim.start()

        return [self.posAnim, self.opacityAnim]

    def hoverCallback(self) -> None:
        """Callback function when the radial menu first sets the button hover to True.

        If a function is not assigned to self._hoverCallback, this is a no op.
        """
        if self._hoverCallback is None:
            return
        else:
            self._hoverCallback()

    def unHoverCallback(self) -> None:
        """Callback function when the radial menu first sets the button hover to False.

        If a function is not assigned to self._unHoverCallback, this is a no op.
        """
        if self._unHoverCallback is None:
            return
        else:
            self._unHoverCallback()


class RadialMenu(QWidget):
    def __init__(self, parent, summonPosition, buttonList: List[ButtonSpecification]):
        super().__init__(parent=parent)
        self.setMouseTracking(True)

        self.setGeometry(self.parent().rect())

        # radius of the cirlce of buttons
        self._buttonCircleRadius = 80

        # radius of the small circle in the middle
        self._innerCircleRadius = 20
        self._buttonList = []
        self._selectedButton = None
        self._mousePressed = False
        self._animFinished = False

        # flag that draws the button circles and mouse line
        self._debugDraw = False

        self._summonPosition = summonPosition
        self._currentMousePos = QPoint(self._summonPosition)

        for button in buttonList:
            self.addButton(button)
        self.setButtonsPositions()

    @property
    def selectedButton(self) -> RadialMenuButton:
        return self._selectedButton

    @selectedButton.setter
    def selectedButton(self, button: RadialMenuButton):
        if button is self._selectedButton:
            return

        # call the unhover callback from the previously selected button
        if self._selectedButton is not None:
            self._selectedButton.unHoverCallback()

        # call the hover callback of the new button
        if button is not None:
            button.hoverCallback()

        # set the new selected button
        self._selectedButton = button

    def clipSummonPosition(self, pos: QPoint) -> QPoint:
        """Clip position of the widget based on the extents of the buttons.

        Parameters
        ----------
        pos : QPoint
            The putative center point of the widget.

        Returns
        -------
        pos : QPoint
            The clipped center point of the widget.
        """
        savepadding = 10
        minSpaceToBorder = savepadding + self._buttonCircleRadius
        maxX = self.rect().width() - minSpaceToBorder
        maxY = self.rect().height() - minSpaceToBorder

        # Guess the button position
        _minX = _minY = _maxX = _maxY = self._buttonList[0]
        for btn in self._buttonList:
            if _minX.pos().x() > btn.pos().x():
                _minX = btn
            if _minY.pos().y() > btn.pos().y():
                _minY = btn
            if _maxX.pos().x() + _maxX.width() < btn.pos().x() + btn.width():
                _maxX = btn
            if _maxY.pos().y() + _maxY.height() < btn.pos().y() + btn.height():
                _maxY = btn

        minX = minSpaceToBorder + _minX.width()
        minY = minSpaceToBorder + _minY.height()
        maxX = maxX - _minX.width()
        maxY = maxY - _maxY.height()

        if pos.x() < minX:
            pos.setX(minX)
        if pos.x() > maxX:
            pos.setX(maxX)
        if pos.y() < minY:
            pos.setY(minY)
        if pos.y() > maxY:
            pos.setY(maxY)
        return pos

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.selectedButton:
            self._mousePressed = True

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.selectedButton:
            self.selectedButton.click()
        self.kill()

    def mouseMoveEvent(self, event):
        self._currentMousePos = event.pos()
        self.update()

    def paintEvent(self, event):
        angle = None
        circleRect = QRect(
            self._summonPosition.x() - self._innerCircleRadius,
            self._summonPosition.y() - self._innerCircleRadius,
            self._innerCircleRadius * 2,
            self._innerCircleRadius * 2,
        )  # The rect of the center circle
        arcSize = 36
        mouseInCircle = (self._currentMousePos.x() - self._summonPosition.x()) ** 2 + (
            self._currentMousePos.y() - self._summonPosition.y()
        ) ** 2 < self._innerCircleRadius ** 2
        bgCirclePen = QPen(QColor("#232323"), 5)
        fgCirclePen = QPen(QColor("#5176b2"), 5)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)
        refLine = QLineF(self._summonPosition, self._currentMousePos)

        # guess the target button
        targetBtn = self._buttonList[0]
        targetLine = QLineF(
            self._summonPosition, targetBtn.pos() + getWidgetCenterPos(targetBtn)
        )
        minAngle = 360
        for btn in self._buttonList:
            # reset the hover and the press effect
            btn.setHover(False)
            btn.setPress(False)

            btnLine = QLineF(self._summonPosition, btn.pos() + getWidgetCenterPos(btn))
            angle = btnLine.angleTo(refLine)
            if angle > 180:
                angle = refLine.angleTo(btnLine)

            if angle < minAngle:
                targetBtn = btn
                targetLine = btnLine  # Used for the debug lines
                minAngle = angle  # Used for the comparison

        if not mouseInCircle:
            normLine = (
                refLine.unitVector()
            )  # Create a line with the same origin and direction but with a length of 1
            angle = QLineF(
                self._summonPosition,
                self._summonPosition + QPoint(self._innerCircleRadius, 0),
            ).angleTo(normLine)

        # Draw Background circle
        painter.setPen(bgCirclePen)
        painter.drawEllipse(circleRect)

        # Draw Forebround circle
        if angle and not mouseInCircle:
            painter.setPen(fgCirclePen)
            painter.drawArc(circleRect, int(angle - arcSize / 2) * 16, arcSize * 16)

        if targetBtn and self._animFinished is True and not mouseInCircle:
            self.selectedButton = targetBtn
            self.selectedButton.setHover(True)
            if self._mousePressed is True:
                self.selectedButton.setPress(True)
        else:
            self.selectedButton = None

        # debug draw
        if self._debugDraw is True:
            painter.setBrush(transparent)
            painter.setPen(Qt.blue)
            painter.drawEllipse(
                self._summonPosition, self._buttonCircleRadius, self._buttonCircleRadius
            )
            painter.drawLine(self._summonPosition, self._currentMousePos)
            for btn in self._buttonList:
                painter.drawLine(
                    self._summonPosition, btn.pos() + getWidgetCenterPos(btn)
                )
            painter.setPen(QPen(QtCore.Qt.yellow, 5))
            painter.drawLine(targetLine)

    def addButton(
        self, buttonSpecification: Union[ButtonSpecification, Dict[str, Any]]
    ) -> None:
        new_button = RadialMenuButton(
            buttonSpecification.name,
            parent=self,
            hoverCallback=buttonSpecification.onHoverTrue,
            unHoverCallback=buttonSpecification.onHoverFalse,
        )
        new_button.clicked.connect(buttonSpecification.onClick)
        new_button.clicked.connect(self.kill)
        self._buttonList.append(new_button)

    def setButtonsPositions(self):
        counter = 0
        for btn in self._buttonList:
            line = QLineF(
                self._summonPosition,
                self._summonPosition + QPoint(self._buttonCircleRadius, 0),
            )
            line.setAngle(counter * (360 / len(self._buttonList)))
            pos = getWidgetCenterPos(btn)
            if abs(int(line.p2().x()) - self._summonPosition.x()) < 3:
                pass
            elif int(line.p2().x()) < self._summonPosition.x():
                pos.setX(btn.rect().width())
            else:
                pos.setX(btn.rect().x())

            if abs(int(line.p2().y()) - self._summonPosition.y()) < 3:
                pass
            elif int(line.p2().y()) < self._summonPosition.y():
                pos.setY(btn.rect().height())
            else:
                pos.setY(btn.rect().y())

            btn.move(line.p2().toPoint() - pos)
            btn.targetPos = line.p2().toPoint()
            counter += 1

    def kill(self):
        self.animGroup.setDirection(QtCore.QAbstractAnimation.Backward)
        self.animGroup.finished.connect(self.hide)
        self.animGroup.start()

    def show(self):
        super().show()

        self._summonPosition = self.clipSummonPosition(self._summonPosition)
        self._currentMousePos = QPoint(self._summonPosition)
        self.setButtonsPositions()

        self.animGroup = QtCore.QParallelAnimationGroup()
        for btn in self._buttonList:
            anims = btn.animate(
                self._summonPosition - getWidgetCenterPos(btn), btn.pos(), False, 100
            )
            for anim in anims:
                self.animGroup.addAnimation(anim)

        self.animGroup.finished.connect(self.animationFinished)

        self.animGroup.start()

    def animationFinished(self):
        self._animFinished = True
        for btn in self._buttonList:
            btn.setEnabled(True)
